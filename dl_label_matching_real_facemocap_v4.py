#!/usr/bin/env python3
"""
FaceMoCap label matching (DL point-wise classifier + Hungarian) with diagnostics visualization.

What was added (high-level):
- Stores per-test-file accuracy + mapping statistics.
- Automatically exports qualitative visualizations for:
  (a) best cases, (b) worst cases, and (c) optional random cases.
- Visualization is portrait-like and consistent across samples using a PCA basis computed
  from the canonical centroids (global reference frame).
- Each saved figure shows:
    1) Ground-truth labeling (colors = true label index after permutation)
    2) Predicted labeling (colors = predicted label)
    3) Correctness overlay (green = correct, red = incorrect, gray = unmatched/invalid)
    4) Optional canonical centroids overlay (small black dots)
- Saves PNGs to --viz_dir (default: viz_label_matching).

Notes:
- This script does NOT change your training/inference logic; it adds diagnostics.
- For headless servers, we force matplotlib "Agg".
"""

import os
import glob
import argparse
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------
# Matplotlib (headless-safe)
# ---------------------------
import matplotlib
matplotlib.use("Agg")  # safe for servers / non-interactive runs
import matplotlib.pyplot as plt


# ============================================================
# Robust CSV loader utilities (handles NaNs / weird tokens)
# ============================================================

def list_csv_files(root: str):
    return sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))


def load_csv_as_sequence(
    filename: str,
    skiprows: int = 5,
    usecols_start: int = 2,
    usecols_end_exclusive: int = 326,  # range(2,326) => 324 cols
    n_points_total: int = 108,
    remove_first_k_points: int = 3,
    max_rows: Optional[int] = None,
) -> np.ndarray:
    """
    Load all rows (frames) from CSV and return seq_pc: (T, 105, 3) float32 with NaNs.
    If max_rows is not None, reads only first max_rows frames (for speed).
    """
    df = pd.read_csv(
        filename,
        skiprows=skiprows,
        header=None,
        usecols=range(usecols_start, usecols_end_exclusive),
        engine="python",
        dtype=str,
        na_values=["", "NA", "NaN", "nan", "None", "null", "NULL"],
        keep_default_na=True,
        nrows=max_rows
    )
    if df.shape[0] < 1:
        raise ValueError("No data rows after skiprows")

    arr = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)  # (T, 324)
    expected = n_points_total * 3
    if arr.shape[1] != expected:
        raise ValueError(f"Expected {expected} columns, got {arr.shape[1]}")

    T = arr.shape[0]
    seq = arr.reshape(T, n_points_total, 3)          # (T, 108, 3)
    seq = seq[:, remove_first_k_points:, :]          # (T, 105, 3)
    return seq


def load_initial_pc(filename: str) -> np.ndarray:
    """
    Load only the first frame as (105,3) with NaNs.
    """
    seq = load_csv_as_sequence(filename, max_rows=1)
    return seq[0]


# ============================================================
# Geometry: masks + Kabsch + Procrustes + centroid ICP
# ============================================================

def valid_mask(pc: np.ndarray) -> np.ndarray:
    return np.isfinite(pc).all(axis=1)


def kabsch_align_masked(source: np.ndarray, target: np.ndarray, mask: np.ndarray):
    idx = np.where(mask)[0]
    if idx.size < 3:
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    A = source[idx].astype(np.float64)
    B = target[idx].astype(np.float64)

    ca = A.mean(axis=0)
    cb = B.mean(axis=0)
    A0 = A - ca
    B0 = B - cb

    H = A0.T @ B0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cb - (R @ ca)
    return R.astype(np.float32), t.astype(np.float32)


def apply_rigid(pc: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ pc.T).T + t


def generalized_procrustes(pcs_train: list[np.ndarray], n_iters: int = 10):
    """
    Build canonical mean shape from ordered training initial frames.
    """
    ref = pcs_train[0].copy()
    for _ in range(n_iters):
        aligned = []
        for pc in pcs_train:
            m = valid_mask(pc) & valid_mask(ref)
            R, t = kabsch_align_masked(pc, ref, m)
            aligned.append(apply_rigid(pc, R, t))
        stack = np.stack(aligned, axis=0)
        new_ref = np.nanmean(stack, axis=0)

        bad = ~np.isfinite(new_ref).any(axis=1)
        new_ref[bad] = ref[bad]
        ref = new_ref

    aligned_final = []
    for pc in pcs_train:
        m = valid_mask(pc) & valid_mask(ref)
        R, t = kabsch_align_masked(pc, ref, m)
        aligned_final.append(apply_rigid(pc, R, t))

    return ref, aligned_final


def compute_centroids(aligned_train: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(aligned_train, axis=0)
    return np.nanmean(stack, axis=0)  # (105,3)


def match_to_centroids(pc_aligned: np.ndarray, centroids: np.ndarray):
    """
    Euclidean cost Hungarian between valid points and valid centroids.
    Returns mapping {point_index -> label}.
    """
    m_pc = valid_mask(pc_aligned)
    m_c = valid_mask(centroids)
    pts_idx = np.where(m_pc)[0]
    lbl_idx = np.where(m_c)[0]
    if pts_idx.size == 0 or lbl_idx.size == 0:
        return {}

    P = pc_aligned[pts_idx]
    C = centroids[lbl_idx]
    cost = np.linalg.norm(P[:, None, :] - C[None, :, :], axis=2)
    row, col = linear_sum_assignment(cost)

    return {int(pts_idx[r]): int(lbl_idx[c]) for r, c in zip(row, col)}


def align_test_cloud_icp_to_centroids(pc: np.ndarray, centroids: np.ndarray, n_iters: int = 7):
    """
    ICP-like alignment without labels:
      - translation init
      - iterate: centroid Hungarian -> Kabsch -> apply
    """
    pc_curr = pc.copy()

    m_pc = valid_mask(pc_curr)
    m_c = valid_mask(centroids)
    if m_pc.any() and m_c.any():
        pc_mean = np.nanmean(pc_curr[m_pc], axis=0)
        c_mean = np.nanmean(centroids[m_c], axis=0)
        pc_curr = pc_curr + (c_mean - pc_mean)

    for _ in range(n_iters):
        pred = match_to_centroids(pc_curr, centroids)
        if len(pred) < 3:
            break
        src_idx = np.array(sorted(pred.keys()), dtype=int)
        dst_lbl = np.array([pred[i] for i in src_idx], dtype=int)
        src = pc_curr[src_idx]
        dst = centroids[dst_lbl]
        m = valid_mask(src) & valid_mask(dst)
        if m.sum() < 3:
            break
        R, t = kabsch_align_masked(src, dst, m)
        pc_curr = apply_rigid(pc_curr, R, t)

    return pc_curr

def align_test_cloud_robust(pc: np.ndarray, centroids: np.ndarray, n_iters: int = 7):
    # 1. Center both clouds
    pc_centered = pc - np.nanmean(pc, axis=0)
    c_centered = centroids - np.nanmean(centroids, axis=0)

    best_pc = None
    best_cost = float('inf')

    # 2. Try 4 rotations around Z axis (assuming Z is up/forward, adjust if needed)
    # Ideally, try rotations around X, Y, and Z if data is very messy.
    # Here is a simple grid search for 4 cardinal directions around the vertical axis.
    rotations = [0, 90, 180, 270]
    
    # You might need to check axes swaps too if your data is very inconsistent.
    # For now, let's assume simple rotation around the 'Up' axis (usually Y or Z).
    # Let's try rotating around X, Y AND Z to be safe (4x3 = 12 checks, fast enough).
    
    transforms = []
    for axis in [0, 1, 2]: # X, Y, Z
        for deg in [0, 90, 180, 270]:
            theta = np.radians(deg)
            c, s = np.cos(theta), np.sin(theta)
            R = np.eye(3)
            if axis == 0: R = np.array([[1,0,0],[0,c,-s],[0,s,c]])
            if axis == 1: R = np.array([[c,0,s],[0,1,0],[-s,0,c]])
            if axis == 2: R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
            transforms.append(R)

    for R_guess in transforms:
        # Apply guess
        pc_guess = (R_guess @ pc_centered.T).T
        
        # Quick check: Mean distance to nearest centroid
        # (We use a cheap heuristic here before running full ICP)
        # Find nearest neighbor for every point
        m_pc = valid_mask(pc_guess)
        if not m_pc.any(): continue
        
        # Simple heuristic: sum of distances to NEAREST centroid
        # (Using a subset of points for speed if needed)
        # Note: This requires computing the distance matrix.
        dists = np.linalg.norm(pc_guess[m_pc][:, None, :] - c_centered[None, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        cost = np.mean(min_dists)

        if cost < best_cost:
            best_cost = cost
            best_pc = pc_guess

    # 3. Now run the existing fine-tuning ICP on the best initialization
    final_pc = align_test_cloud_icp_to_centroids(best_pc, centroids, n_iters)
    return final_pc
# ============================================================
# DL model: point-wise MLP classifier
# ============================================================

class PointMLPClassifier(nn.Module):
    def __init__(self, in_dim=3, hidden=128, num_classes=105):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        # x: (B,3)
        return self.net(x)


def standardize_points(pc: np.ndarray):
    """
    Simple per-cloud normalization in canonical space:
      - subtract mean of valid points
      - divide by robust scale (median norm)
    """
    m = valid_mask(pc)
    if not m.any():
        return pc
    X = pc.copy()
    mu = np.nanmean(X[m], axis=0)
    X[m] = X[m] - mu
    scale = np.nanmedian(np.linalg.norm(X[m], axis=1))
    if np.isfinite(scale) and scale > 1e-6:
        X[m] = X[m] / scale
    return X


# ============================================================
# Training data sampler (uses ordered training files)
# ============================================================

def sample_training_batch(
    rng: np.random.Generator,
    train_files: list[str],
    mean_shape: np.ndarray,
    use_all_frames: bool,
    frames_per_file: int,
    points_per_frame: int,
    max_rows_per_file: Optional[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns batch (X, y) where:
      X: (B,3) float32
      y: (B,) int64
    """
    X_list = []
    y_list = []
    n_labels = 105

    n_files = 6 if use_all_frames else 12
    chosen = rng.choice(train_files, size=min(n_files, len(train_files)), replace=False)

    for f in chosen:
        try:
            if use_all_frames:
                seq = load_csv_as_sequence(f, max_rows=max_rows_per_file)  # (T,105,3)
                T = seq.shape[0]
                if T == 0:
                    continue
                fr_idx = rng.integers(0, T, size=min(frames_per_file, T))
                for ti in fr_idx:
                    pc = seq[ti]
                    m = valid_mask(pc) & valid_mask(mean_shape)
                    R, t = kabsch_align_masked(pc, mean_shape, m)
                    pc_al = apply_rigid(pc, R, t)
                    pc_al = standardize_points(pc_al)

                    pts = rng.integers(0, n_labels, size=points_per_frame)
                    for pidx in pts:
                        if np.isfinite(pc_al[pidx]).all():
                            X_list.append(pc_al[pidx])
                            y_list.append(pidx)
            else:
                pc = load_initial_pc(f)
                m = valid_mask(pc) & valid_mask(mean_shape)
                R, t = kabsch_align_masked(pc, mean_shape, m)
                pc_al = apply_rigid(pc, R, t)
                pc_al = standardize_points(pc_al)

                pts = rng.integers(0, n_labels, size=points_per_frame)
                for pidx in pts:
                    if np.isfinite(pc_al[pidx]).all():
                        X_list.append(pc_al[pidx])
                        y_list.append(pidx)

        except Exception:
            continue

    if len(X_list) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.int64)

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y


# ============================================================
# Inference: Hungarian using -log p(label)
# ============================================================

def hungarian_from_probs(probs: np.ndarray, valid_points_mask: np.ndarray):
    """
    probs: (105, 105) probability per point index (row) for each label (col)
    valid_points_mask: (105,) whether that point row is a valid observed point
    Returns mapping {point_index -> label}.
    """
    pts_idx = np.where(valid_points_mask)[0]
    if pts_idx.size == 0:
        return {}

    P = probs[pts_idx]  # (m,105)
    eps = 1e-12
    cost = -np.log(P + eps)

    row, col = linear_sum_assignment(cost)
    return {int(pts_idx[r]): int(c) for r, c in zip(row, col)}


def evaluate_mapping(true_labels_per_row: np.ndarray, pred_map: dict[int, int]) -> tuple[float, int]:
    if len(pred_map) == 0:
        return 0.0, 0
    correct = 0
    for i, lab in pred_map.items():
        correct += int(true_labels_per_row[i] == lab)
    return correct / len(pred_map), len(pred_map)


# ============================================================
# Visualization helpers (portrait-like, consistent axes)
# ============================================================

def compute_pca_basis_from_centroids(centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (mu, R) where:
      - mu is centroid mean (3,)
      - R is 3x3 basis matrix whose columns are principal axes (e1,e2,e3).
    We use SVD on centered valid centroids.
    """
    m = valid_mask(centroids)
    X = centroids[m].astype(np.float64)
    mu = X.mean(axis=0)
    X0 = X - mu
    _, _, Vt = np.linalg.svd(X0, full_matrices=False)
    # Vt rows are principal directions; we want columns as basis vectors
    e1 = Vt[0]
    e2 = Vt[1]
    # ensure right-handed frame
    e3 = np.cross(e1, e2)
    e3_norm = np.linalg.norm(e3)
    if e3_norm < 1e-9:
        e3 = Vt[2]
    else:
        e3 = e3 / e3_norm
    # Re-orthogonalize e2 to ensure numerical stability
    e2 = np.cross(e3, e1)
    e1 = e1 / (np.linalg.norm(e1) + 1e-12)
    e2 = e2 / (np.linalg.norm(e2) + 1e-12)
    R = np.stack([e1, e2, e3], axis=1).astype(np.float32)  # 3x3
    return mu.astype(np.float32), R


def project_to_portrait_2d(pc: np.ndarray, mu: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Projects 3D points to 2D using PCA basis (e1,e2) -> "portrait" plane.
    """
    X = pc.copy()
    m = valid_mask(X)
    if m.any():
        X[m] = X[m] - mu[None, :]
        # coordinates in PCA basis
        Y = (R.T @ X[m].T).T  # (n,3)
        out = np.full((pc.shape[0], 2), np.nan, dtype=np.float32)
        out[m] = Y[:, :2]
        return out
    return np.full((pc.shape[0], 2), np.nan, dtype=np.float32)


def _axis_equal_2d(ax):
    ax.set_aspect("equal", adjustable="box")


def _scatter_labels_2d(ax, xy: np.ndarray, labels: np.ndarray, title: str,
                       cmap_name: str = "turbo", s: float = 14.0, alpha: float = 0.95):
    m = np.isfinite(xy).all(axis=1)
    if not m.any():
        ax.set_title(title + " (no valid points)")
        ax.axis("off")
        return

    cm = plt.get_cmap(cmap_name)
    # labels should be in [0,104] or -1 for invalid/unmatched
    lbl = labels.copy()
    # Map -1 to a dedicated color (light gray) by masking
    m_valid_lbl = (lbl >= 0)
    # Plot unmatched/invalid first
    m_un = m & (~m_valid_lbl)
    if m_un.any():
        ax.scatter(xy[m_un, 0], xy[m_un, 1], s=s, c="lightgray", alpha=0.7, linewidths=0)

    m_ok = m & m_valid_lbl
    if m_ok.any():
        c = cm(lbl[m_ok] / 104.0)
        ax.scatter(xy[m_ok, 0], xy[m_ok, 1], s=s, c=c, alpha=alpha, linewidths=0)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    _axis_equal_2d(ax)


def _scatter_correctness_2d(ax, xy: np.ndarray, correctness: np.ndarray, title: str,
                           s: float = 16.0, alpha: float = 0.95):
    """
    correctness: -1 (unmatched/invalid), 0 (wrong), 1 (correct)
    """
    m = np.isfinite(xy).all(axis=1)
    if not m.any():
        ax.set_title(title + " (no valid points)")
        ax.axis("off")
        return

    m_un = m & (correctness < 0)
    m_ok = m & (correctness == 1)
    m_bad = m & (correctness == 0)

    if m_un.any():
        ax.scatter(xy[m_un, 0], xy[m_un, 1], s=s, c="lightgray", alpha=0.7, linewidths=0)
    if m_ok.any():
        ax.scatter(xy[m_ok, 0], xy[m_ok, 1], s=s, c="green", alpha=alpha, linewidths=0)
    if m_bad.any():
        ax.scatter(xy[m_bad, 0], xy[m_bad, 1], s=s, c="red", alpha=alpha, linewidths=0)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    _axis_equal_2d(ax)


def save_diagnostic_figure(
    out_path: str,
    pc_aligned_std: np.ndarray,
    centroids_std: np.ndarray,
    labels_true_per_row: np.ndarray,
    pred_map: dict[int, int],
    pca_mu: np.ndarray,
    pca_R: np.ndarray,
    title: str,
    show_centroids: bool = True,
    dpi: int = 160,
):
    """
    Creates a 1x3 large subplot:
      (1) GT labels
      (2) Predicted labels
      (3) Correctness
    with optional centroid overlay.
    """
    # Build predicted label per row (-1 for invalid/unmatched)
    pred_labels = np.full((105,), -1, dtype=int)
    for i, lab in pred_map.items():
        pred_labels[i] = int(lab)

    # Correctness per row: -1 (invalid/unmatched), 0 wrong, 1 correct
    correctness = np.full((105,), -1, dtype=int)
    vm = valid_mask(pc_aligned_std)
    for i in range(105):
        if not vm[i]:
            continue
        if i not in pred_map:
            continue
        correctness[i] = int(labels_true_per_row[i] == pred_map[i])

    # Project to portrait 2D
    xy = project_to_portrait_2d(pc_aligned_std, pca_mu, pca_R)
    xy_c = project_to_portrait_2d(centroids_std, pca_mu, pca_R) if show_centroids else None

    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    _scatter_labels_2d(ax1, xy, labels_true_per_row, "Ground truth labels (color = label index)")
    _scatter_labels_2d(ax2, xy, pred_labels, "Predicted labels (color = assigned label)")
    _scatter_correctness_2d(ax3, xy, correctness, "Correctness (green=correct, red=wrong)")

    # Overlay centroids as small black points for reference
    if show_centroids and (xy_c is not None):
        mc = np.isfinite(xy_c).all(axis=1)
        for ax in (ax1, ax2, ax3):
            if mc.any():
                ax.scatter(xy_c[mc, 0], xy_c[mc, 1], s=8, c="black", alpha=0.45, linewidths=0)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ============================================================
# Diagnostics container
# ============================================================

@dataclass
class TestCaseResult:
    filepath: str
    acc: float
    nmatch: int
    pc_perm: np.ndarray
    pc_aligned_std: np.ndarray
    labels_true_per_row: np.ndarray
    pred_map: dict[int, int]


# ============================================================
# Reporting: detailed statistical analysis
# ============================================================

def write_analysis_report(
    out_path: str,
    cases: list[TestCaseResult],
    centroids: np.ndarray,
    pca_mu: np.ndarray,
    pca_R: np.ndarray,
):
    """
    Failure-oriented report based ONLY on observable quantities.
    No probabilities are assumed available.
    """

    def classify(acc):
        if acc > 0.95:
            return "EXCELLENT"
        elif acc >= 0.80:
            return "REGULAR"
        elif acc >= 0.10:
            return "BAD"
        else:
            return "TERRIBLE"

    def subject_id(path):
        return os.path.basename(path).split("_")[0]

    groups = {k: [] for k in ["EXCELLENT", "REGULAR", "BAD", "TERRIBLE"]}
    for c in cases:
        groups[classify(c.acc)].append(c)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("FACE MOCAP LABEL MATCHING – STRUCTURAL FAILURE ANALYSIS REPORT\n")
        f.write("=" * 100 + "\n\n")

        f.write("Accuracy class definitions:\n")
        f.write("  EXCELLENT : acc > 95%\n")
        f.write("  REGULAR   : 95% ≥ acc ≥ 80%\n")
        f.write("  BAD       : 80% > acc ≥ 10%\n")
        f.write("  TERRIBLE  : acc < 10%\n\n")

        for cls, cls_cases in groups.items():
            f.write("=" * 100 + "\n")
            f.write(f"{cls} CASES (n={len(cls_cases)})\n")
            f.write("=" * 100 + "\n")

            if not cls_cases:
                f.write("No samples in this category.\n\n")
                continue

            accs = np.array([c.acc for c in cls_cases])
            nmatches = np.array([c.nmatch for c in cls_cases])

            f.write(
                f"Accuracy: mean={accs.mean()*100:.2f}% | "
                f"median={np.median(accs)*100:.2f}% | "
                f"min={accs.min()*100:.2f}% | max={accs.max()*100:.2f}%\n"
            )
            f.write(
                f"Matched points: mean={nmatches.mean():.1f}/105 | "
                f"min={nmatches.min()} | max={nmatches.max()}\n\n"
            )

            # --------------------------------------------------
            # Geometric sanity check
            # --------------------------------------------------
            all_pts = []
            for c in cls_cases:
                vm = valid_mask(c.pc_aligned_std)
                all_pts.append(c.pc_aligned_std[vm])
            all_pts = np.vstack(all_pts)
            norms = np.linalg.norm(all_pts, axis=1)

            f.write("Geometric statistics (canonical space):\n")
            f.write(f"  Mean XYZ: {all_pts.mean(axis=0)}\n")
            f.write(f"  Std  XYZ: {all_pts.std(axis=0)}\n")
            f.write(f"  Mean norm: {norms.mean():.4f} | Std norm: {norms.std():.4f}\n\n")

            # --------------------------------------------------
            # NEW 1: Spatial correctness vs wrongness
            # --------------------------------------------------
            dist_true = []
            dist_pred = []

            for c in cls_cases:
                for i, lab_pred in c.pred_map.items():
                    lab_true = c.labels_true_per_row[i]
                    p = c.pc_aligned_std[i]
                    if not np.isfinite(p).all():
                        continue
                    dist_true.append(
                        np.linalg.norm(p - centroids[lab_true])
                    )
                    dist_pred.append(
                        np.linalg.norm(p - centroids[lab_pred])
                    )

            dist_true = np.array(dist_true)
            dist_pred = np.array(dist_pred)

            f.write("Spatial consistency diagnostics:\n")
            f.write(
                f"  Dist to TRUE centroid: mean={dist_true.mean():.4f} | "
                f"std={dist_true.std():.4f}\n"
            )
            f.write(
                f"  Dist to PRED centroid: mean={dist_pred.mean():.4f} | "
                f"std={dist_pred.std():.4f}\n\n"
            )

            # --------------------------------------------------
            # NEW 2: Permutation cycle accuracy breakdown
            # --------------------------------------------------
            fixed_points = []
            largest_cycles = []
            cycle_accs = []

            for c in cls_cases:
                perm = c.pred_map
                visited = set()
                cycles = []

                for i in perm:
                    if i in visited:
                        continue
                    cycle = []
                    j = i
                    while j not in visited and j in perm:
                        visited.add(j)
                        cycle.append(j)
                        j = perm[j]
                    if cycle:
                        cycles.append(cycle)

                fixed = sum(1 for cy in cycles if len(cy) == 1)
                fixed_points.append(fixed)
                largest_cycles.append(max(len(cy) for cy in cycles))

                for cy in cycles:
                    correct = sum(
                        1 for i in cy
                        if c.labels_true_per_row[i] == perm.get(i, -1)
                    )
                    cycle_accs.append(correct / len(cy))

            f.write("Permutation structure diagnostics:\n")
            f.write(
                f"  Fixed points: mean={np.mean(fixed_points):.1f} | "
                f"min={np.min(fixed_points)} | max={np.max(fixed_points)}\n"
            )
            f.write(
                f"  Largest cycle size: mean={np.mean(largest_cycles):.1f} | "
                f"min={np.min(largest_cycles)} | max={np.max(largest_cycles)}\n"
            )
            f.write(
                f"  Cycle-level accuracy: mean={np.mean(cycle_accs)*100:.2f}% | "
                f"median={np.median(cycle_accs)*100:.2f}%\n\n"
            )

            # --------------------------------------------------
            # NEW 3: Missing data
            # --------------------------------------------------
            missing_counts = []
            for c in cls_cases:
                missing_counts.append((~valid_mask(c.pc_aligned_std)).sum())

            f.write("Missing data diagnostics:\n")
            f.write(
                f"  Missing points per cloud: "
                f"mean={np.mean(missing_counts):.1f} | "
                f"min={np.min(missing_counts)} | max={np.max(missing_counts)}\n\n"
            )

            # --------------------------------------------------
            # NEW 4: Subject-level aggregation
            # --------------------------------------------------
            subj_hist = {}
            for c in cls_cases:
                sid = subject_id(c.filepath)
                subj_hist[sid] = subj_hist.get(sid, 0) + 1

            f.write("Subject-level occurrence:\n")
            for sid, cnt in sorted(subj_hist.items(), key=lambda x: -x[1]):
                f.write(f"  {sid:15s}: {cnt} samples\n")

            f.write("\nDetailed case listing:\n")
            for c in sorted(cls_cases, key=lambda x: x.acc):
                f.write(
                    f"  {os.path.basename(c.filepath):50s} | "
                    f"acc={c.acc*100:6.2f}% | matched={c.nmatch:3d}/105\n"
                )

            f.write("\n")

    print(f"Revised structural report written to: {out_path}")

# ============================================================
# Main experiment
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to .../Data_FaceMoCap/Sujets_Sains")
    ap.add_argument("--train_ratio", type=float, default=0.9, choices=[0.8, 0.9])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpa_iters", type=int, default=10)

    # DL training
    ap.add_argument("--use_all_frames_train", action="store_true",
                    help="If set, sample multiple frames per training CSV (recommended).")
    ap.add_argument("--frames_per_file", type=int, default=8,
                    help="How many frames to sample per chosen file per batch (when using all frames).")
    ap.add_argument("--points_per_frame", type=int, default=32,
                    help="How many point indices to sample per frame for training.")
    ap.add_argument("--max_rows_per_file", type=int, default=250,
                    help="Cap number of rows read per file when using all frames (speed control).")

    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--steps_per_epoch", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)

    # test alignment
    ap.add_argument("--icp_iters", type=int, default=7)

    # model saving
    ap.add_argument("--save_model", type=str, default="models/facemocap_hardalign_model.pth",
                    help="Path to save the trained model and artifacts.")

    # evaluation mode
    ap.add_argument("--eval_only", action="store_true",
                    help="Skip training; load checkpoint and evaluate on test set.")

    # visualization
    ap.add_argument("--viz_dir", type=str, default="viz_label_matching",
                    help="Directory where diagnostic images are written.")
    ap.add_argument("--viz_topk", type=int, default=12,
                    help="How many best and worst cases to export.")
    ap.add_argument("--viz_random", type=int, default=0,
                    help="Additionally export N random test cases.")
    ap.add_argument("--viz_show_centroids", action="store_true",
                    help="Overlay canonical centroids (in black) on all panels.")
    ap.add_argument("--viz_dpi", type=int, default=160)

    # reporting
    ap.add_argument("--report", type=str, default="",
                    help="If set, write detailed analysis report to this .txt file.")

    args = ap.parse_args()

    # Validate eval_only mode
    if args.eval_only:
        if not os.path.exists(args.save_model):
            raise SystemExit(f"--eval_only requires existing checkpoint: {args.save_model}")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    files = list_csv_files(args.root)
    if len(files) == 0:
        raise SystemExit(f"No CSV files found under: {args.root}")

    # Filter usable files by attempting to load initial frame (cheap)
    usable = []
    for f in files:
        try:
            pc = load_initial_pc(f)
            if pc.shape == (105, 3):
                usable.append(f)
        except Exception:
            continue

    if len(usable) < 20:
        raise SystemExit(f"Too few usable CSVs: {len(usable)}")

    print(f"Loaded {len(usable)} usable CSVs.")

    idx = np.arange(len(usable))
    rng.shuffle(idx)
    n_train = int(round(args.train_ratio * len(usable)))
    train_files = [usable[i] for i in idx[:n_train]]
    test_files = [usable[i] for i in idx[n_train:]]

    print(f"Train clouds (files): {len(train_files)} | Test clouds (files): {len(test_files)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"cuda available: {torch.cuda.is_available()} | device: {device}")

    if args.eval_only:
        # Load checkpoint
        print(f"\nLoading checkpoint from: {args.save_model}")
        state = torch.load(args.save_model, map_location=device, weights_only=False)
        centroids = state["centroids"]
        mean_shape = state["mean_shape"]
        pca_mu, pca_R = compute_pca_basis_from_centroids(centroids)
        centroids_std = standardize_points(centroids.copy())

        model = PointMLPClassifier(in_dim=3, hidden=args.hidden, num_classes=105).to(device)
        model.load_state_dict(state["model_state_dict"])
        print("Model weights loaded successfully.")
    else:
        # Build canonical mean shape from training initial frames (ordered)
        pcs_train_init = [load_initial_pc(f) for f in train_files]
        mean_shape, aligned_train_init = generalized_procrustes(pcs_train_init, n_iters=args.gpa_iters)
        centroids = compute_centroids(aligned_train_init)

        # Precompute PCA basis for consistent "portrait" views
        pca_mu, pca_R = compute_pca_basis_from_centroids(centroids)

        # For visualization, we want centroids in the same standardized space as test samples
        # (so overlays are meaningful).
        centroids_std = standardize_points(centroids.copy())

        # Train point-wise classifier in canonical space
        model = PointMLPClassifier(in_dim=3, hidden=args.hidden, num_classes=105).to(device)
        opt = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for ep in range(args.epochs):
            losses = []
            for _step in range(args.steps_per_epoch):
                X, y = sample_training_batch(
                    rng=rng,
                    train_files=train_files,
                    mean_shape=mean_shape,
                    use_all_frames=args.use_all_frames_train,
                    frames_per_file=args.frames_per_file,
                    points_per_frame=args.points_per_frame,
                    max_rows_per_file=args.max_rows_per_file if args.use_all_frames_train else None,
                )
                if X.shape[0] == 0:
                    continue

                xb = torch.from_numpy(X).to(device)
                yb = torch.from_numpy(y).to(device)

                opt.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))

            mean_loss = float(np.mean(losses)) if losses else float("nan")
            print(f"Epoch {ep+1:02d}/{args.epochs} | loss={mean_loss:.4f}")

        # Save trained model + useful artifacts
        try:
            save_path = args.save_model
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "centroids": centroids.astype(np.float32),
                "mean_shape": mean_shape.astype(np.float32),
                "args": vars(args),
            }
            torch.save(state, save_path)
            print(f"Saved trained model and artifacts to: {args.save_model}")
        except Exception as e:
            print(f"Warning: failed to save model -> {e}")

    # Evaluate on test set (initial frames only), with permutation recovery
    model.eval()
    base_labels = np.arange(105, dtype=int)

    accs = []
    matched = []
    cases: list[TestCaseResult] = []

    for f in test_files:
        pc0 = load_initial_pc(f)

        # permute (unknown order), store ground truth
        perm = rng.permutation(105)
        pc_perm = pc0[perm]
        labels_perm = base_labels[perm]  # true label per row

        # align without labels (ICP to centroids)
        pc_al = align_test_cloud_robust(pc_perm, centroids, n_iters=args.icp_iters)
        pc_al = standardize_points(pc_al)

        # predict per-point probabilities
        vm = valid_mask(pc_al)
        X = np.zeros((105, 3), dtype=np.float32)
        X[vm] = pc_al[vm]
        xb = torch.from_numpy(X).to(device)

        with torch.no_grad():
            logits = model(xb)  # (105,105)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        # Hungarian using -log probs (one-to-one)
        pred_map = hungarian_from_probs(probs, vm)
        acc, nmatch = evaluate_mapping(labels_perm, pred_map)

        accs.append(acc)
        matched.append(nmatch)

        cases.append(TestCaseResult(
            filepath=f,
            acc=acc,
            nmatch=nmatch,
            pc_perm=pc_perm,
            pc_aligned_std=pc_al,
            labels_true_per_row=labels_perm,
            pred_map=pred_map
        ))

    accs_arr = np.array(accs, dtype=float)
    matched_arr = np.array(matched, dtype=int)

    print("\n=== Results (real data, DL embeddings + Hungarian) ===")
    print(f"Mean accuracy:   {accs_arr.mean()*100:.2f}%")
    print(f"Median accuracy: {np.median(accs_arr)*100:.2f}%")
    print(f"Min / Max:       {accs_arr.min()*100:.2f}% / {accs_arr.max()*100:.2f}%")
    print(f"Mean matched pts per cloud: {matched_arr.mean():.1f} / 105 (excluding NaNs)")

    # ------------------------------------------------------------
    # Export diagnostic visualizations: best + worst (+ optional random)
    # ------------------------------------------------------------
    if len(cases) > 0:
        os.makedirs(args.viz_dir, exist_ok=True)

        # sort by accuracy, then by matched points (secondary)
        order = sorted(range(len(cases)), key=lambda i: (cases[i].acc, cases[i].nmatch))
        k = int(max(0, args.viz_topk))

        worst_idx = order[:min(k, len(order))]
        best_idx = order[-min(k, len(order)):] if k > 0 else []
        best_idx = list(reversed(best_idx))  # highest first

        # optional random subset
        rand_idx = []
        if args.viz_random > 0:
            rr = rng.choice(len(cases), size=min(args.viz_random, len(cases)), replace=False)
            rand_idx = [int(x) for x in rr]

        def export_group(group_name: str, indices: list[int]):
            if len(indices) == 0:
                return
            group_dir = os.path.join(args.viz_dir, group_name)
            os.makedirs(group_dir, exist_ok=True)

            for rank, i in enumerate(indices, start=1):
                c = cases[i]
                base = os.path.splitext(os.path.basename(c.filepath))[0]
                # keep filename informative and stable
                fname = f"{rank:02d}_acc_{c.acc*100:06.2f}_match_{c.nmatch:03d}_{base}.png"
                out_path = os.path.join(group_dir, fname)

                title = (
                    f"{group_name.upper()} | acc={c.acc*100:.2f}% | matched={c.nmatch}/105\n"
                    f"file: {c.filepath}"
                )

                save_diagnostic_figure(
                    out_path=out_path,
                    pc_aligned_std=c.pc_aligned_std,
                    centroids_std=centroids_std,
                    labels_true_per_row=c.labels_true_per_row,
                    pred_map=c.pred_map,
                    pca_mu=pca_mu,
                    pca_R=pca_R,
                    title=title,
                    show_centroids=args.viz_show_centroids,
                    dpi=args.viz_dpi,
                )

        export_group("worst", worst_idx)
        export_group("best", best_idx)
        export_group("random", rand_idx)

        print(f"\nSaved diagnostic visualizations to: {args.viz_dir}")
        if k > 0:
            print(f"- worst/{min(k, len(order))} cases and best/{min(k, len(order))} cases exported")
        if args.viz_random > 0:
            print(f"- random/{min(args.viz_random, len(cases))} cases exported")

    # Write detailed analysis report (success vs failure)
    if args.report and len(cases) > 0:
        write_analysis_report(
            out_path=args.report,
            cases=cases,
            centroids=centroids,
            pca_mu=pca_mu,
            pca_R=pca_R,
        )

    print("Done.")


if __name__ == "__main__":
    main()