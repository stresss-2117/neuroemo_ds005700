"""
ReHo (Regional Homogeneity) Analysis for NeuroEmo ds005700.

Pipeline:
  1. Compute KCC-based ReHo map per subject/task (26-neighbor cube)
  2. Normalize ReHo maps (z-score within brain mask)
  3. Group-level one-sample t-test (ReHo > 0) per task
  4. Paired t-test: rest vs. emotion task
  5. ROI-level ReHo extraction (AAL atlas)
  6. Visualize: individual maps, group maps, ROI bar charts, rest-vs-fe comparison

Usage:
  python 04_reho_analysis.py [--subjects sub-01 ...] [--tasks rest fe] [--n-jobs 1]
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import image, plotting
from nilearn.datasets import load_mni152_brain_mask
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
PREPROC_DIR = BASE_DIR / "preprocessed"
REHO_DIR = BASE_DIR / "reho_analysis"
FIG_DIR = BASE_DIR / "outputs" / "figures"
LOG_DIR = BASE_DIR / "outputs" / "logs"

for d in [REHO_DIR, FIG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────
_sh = logging.StreamHandler(sys.stdout)
_sh.stream = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[_sh, logging.FileHandler(LOG_DIR / "04_reho.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

# ── Parameters ─────────────────────────────────────────────────────────────
NEIGHBOR_MODE = 26      # 6, 18, or 26 neighbors
P_THRESHOLD = 0.05      # uncorrected p-value for group maps
Z_THRESHOLD = 2.3       # z-score threshold for display


# ═══════════════════════════════════════════════════════════════════════════
# Core ReHo computation — Kendall's Coefficient of Concordance (KCC)
# ═══════════════════════════════════════════════════════════════════════════

def get_neighbor_offsets(mode: int = 26) -> np.ndarray:
    """Return voxel offsets for the specified neighborhood (6/18/26)."""
    offsets = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                dist = abs(dx) + abs(dy) + abs(dz)
                if mode == 6 and dist == 1:
                    offsets.append((dx, dy, dz))
                elif mode == 18 and dist <= 2:
                    offsets.append((dx, dy, dz))
                elif mode == 26:
                    offsets.append((dx, dy, dz))
    return np.array(offsets)


def compute_reho(img: nib.Nifti1Image, mask_data: np.ndarray,
                 mode: int = NEIGHBOR_MODE) -> nib.Nifti1Image:
    """
    Vectorized KCC-based ReHo computation.

    Key insight: KCC = 12*S / (k^2*(T^3-T)) where
      S = sum_t (R_t - mean_R)^2,  R_t = sum of ranks at time t across k neighbors,
      mean_R = k*(T+1)/2.

    Implementation:
      1. Rank every voxel's time series independently (vectorized over voxels).
      2. For each of the 26 neighbor offsets, shift the rank volume and accumulate
         into a running rank-sum array — pure numpy, no Python voxel loop.
      3. Compute KCC analytically from the accumulated rank sums.
    """
    data = img.get_fdata(dtype=np.float32)   # (X, Y, Z, T)
    X, Y, Z, T = data.shape
    offsets = get_neighbor_offsets(mode)
    n_in_mask = int(mask_data.sum())
    log.info(f"  Ranking {n_in_mask} voxels time-series …")

    # Step 1: rank all voxels at once — reshape to (N_vox, T), rank each row
    flat = data.reshape(-1, T)
    flat_mask = mask_data.flatten()
    flat_ranks = np.zeros_like(flat)
    in_mask_idx = np.where(flat_mask)[0]
    for i in in_mask_idx:
        flat_ranks[i] = stats.rankdata(flat[i])
    ranks = flat_ranks.reshape(X, Y, Z, T)   # (X, Y, Z, T)

    log.info(f"  Accumulating neighbor rank sums ({mode} offsets) …")

    # Step 2: accumulate rank sums and neighbor counts via array shifts
    rank_sum = ranks.copy()                               # center voxel always included
    neighbor_count = mask_data.astype(np.float32)        # 1 for center

    for dx, dy, dz in offsets:
        # Source slice: ranks[x+dx, y+dy, z+dz]  (neighbor)
        # Target slice: rank_sum[x, y, z]         (center)
        sx = slice(max(0,  dx), X + min(0,  dx))
        sy = slice(max(0,  dy), Y + min(0,  dy))
        sz = slice(max(0,  dz), Z + min(0,  dz))
        tx = slice(max(0, -dx), X + min(0, -dx))
        ty = slice(max(0, -dy), Y + min(0, -dy))
        tz = slice(max(0, -dz), Z + min(0, -dz))

        nb_mask = mask_data[sx, sy, sz]
        ct_mask = mask_data[tx, ty, tz]
        valid = (nb_mask & ct_mask).astype(np.float32)   # (sub_X, sub_Y, sub_Z)

        rank_sum[tx, ty, tz] += ranks[sx, sy, sz] * valid[..., np.newaxis]
        neighbor_count[tx, ty, tz] += valid

    # Step 3: compute KCC analytically
    k = neighbor_count                                    # (X, Y, Z)
    mean_R = k * (T + 1) / 2.0                           # expected mean rank sum
    S = np.sum((rank_sum - mean_R[..., np.newaxis]) ** 2, axis=3)
    denom = k ** 2 * (T ** 3 - T)

    with np.errstate(divide="ignore", invalid="ignore"):
        reho_map = np.where(denom > 0, 12.0 * S / denom, 0.0).astype(np.float32)
    reho_map[~mask_data.astype(bool)] = 0.0

    log.info(f"  ReHo computed  in-brain range [{reho_map[mask_data.astype(bool)].min():.4f}, "
             f"{reho_map[mask_data.astype(bool)].max():.4f}]")
    return nib.Nifti1Image(reho_map, img.affine, img.header)


# ═══════════════════════════════════════════════════════════════════════════
# Normalization
# ═══════════════════════════════════════════════════════════════════════════

def normalize_reho(reho_img: nib.Nifti1Image,
                   mask_img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Z-score ReHo within the brain mask:
      z = (ReHo - mean_brain) / std_brain
    This removes global scanner/session effects.
    """
    reho_data = reho_img.get_fdata(dtype=np.float32)
    mask_data = mask_img.get_fdata().astype(bool)

    brain_vals = reho_data[mask_data]
    mu = brain_vals.mean()
    sigma = brain_vals.std()

    z_data = np.zeros_like(reho_data)
    z_data[mask_data] = (brain_vals - mu) / (sigma + 1e-12)

    log.info(f"  Normalized ReHo: mean={mu:.4f}  std={sigma:.4f}")
    return nib.Nifti1Image(z_data, reho_img.affine, reho_img.header)


# ═══════════════════════════════════════════════════════════════════════════
# ROI extraction
# ═══════════════════════════════════════════════════════════════════════════

def _load_atlas():
    """
    Try atlases in order until one downloads successfully.
    Returns (maps_img, labels) or raises RuntimeError.
    """
    import ssl
    # Temporarily disable SSL verification for atlas downloads on Windows
    _orig_ctx = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context

    try:
        try:
            from nilearn.datasets import fetch_atlas_aal
            atlas = fetch_atlas_aal(version="SPM12")
            return atlas.maps, atlas.labels
        except Exception:
            pass
        try:
            from nilearn.datasets import fetch_atlas_harvard_oxford
            atlas = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
            return atlas.maps, atlas.labels
        except Exception:
            pass
        try:
            from nilearn.datasets import fetch_atlas_destrieux_2009
            atlas = fetch_atlas_destrieux_2009()
            return atlas.maps, atlas.labels
        except Exception:
            pass
        raise RuntimeError("All atlas downloads failed")
    finally:
        ssl._create_default_https_context = _orig_ctx


def extract_roi_values(reho_img: nib.Nifti1Image) -> dict:
    """Extract mean ReHo per atlas region."""
    maps_img, labels = _load_atlas()
    masker = NiftiLabelsMasker(
        labels_img=maps_img,
        labels=labels,
        resampling_target="data",
        standardize=False,
    )
    # NiftiLabelsMasker needs 4-D input
    img_4d = image.new_img_like(reho_img, reho_img.get_fdata()[..., np.newaxis])
    roi_vals = masker.fit_transform(img_4d)
    return dict(zip(labels, roi_vals[0]))


# ═══════════════════════════════════════════════════════════════════════════
# Group statistics
# ═══════════════════════════════════════════════════════════════════════════

def group_one_sample_t(reho_imgs: list, mask_img: nib.Nifti1Image,
                        subject_ids: list) -> tuple:
    """
    One-sample t-test: H0: mean ReHo = 0, across subjects.
    Returns (t_img, p_img, mean_img).
    """
    ref = reho_imgs[0]
    mask_data = mask_img.get_fdata().astype(bool)

    # Stack all subjects into (n_subjects, X, Y, Z)
    stacked = np.stack(
        [image.resample_to_img(img, ref, interpolation="continuous").get_fdata(dtype=np.float32)
         for img in reho_imgs],
        axis=0,
    )

    t_map = np.zeros(ref.shape[:3], dtype=np.float32)
    p_map = np.ones(ref.shape[:3], dtype=np.float32)
    mean_map = stacked.mean(axis=0)

    in_mask = np.argwhere(mask_data)
    log.info(f"  Running one-sample t-test across {len(reho_imgs)} subjects …")

    for x, y, z in in_mask:
        vals = stacked[:, x, y, z]
        if vals.std() < 1e-10:
            continue
        t, p = stats.ttest_1samp(vals, popmean=0)
        t_map[x, y, z] = t
        p_map[x, y, z] = p

    t_img = nib.Nifti1Image(t_map, ref.affine)
    p_img = nib.Nifti1Image(p_map, ref.affine)
    mean_img = nib.Nifti1Image(mean_map, ref.affine)
    return t_img, p_img, mean_img


def group_paired_t(reho_rest: list, reho_fe: list,
                   mask_img: nib.Nifti1Image) -> tuple:
    """
    Paired t-test: rest vs. emotion task ReHo.
    Returns (t_img, p_img, diff_img).
    """
    ref = reho_rest[0]
    mask_data = mask_img.get_fdata().astype(bool)

    stack_rest = np.stack(
        [image.resample_to_img(img, ref, interpolation="continuous").get_fdata(dtype=np.float32)
         for img in reho_rest], axis=0)
    stack_fe = np.stack(
        [image.resample_to_img(img, ref, interpolation="continuous").get_fdata(dtype=np.float32)
         for img in reho_fe], axis=0)

    t_map = np.zeros(ref.shape[:3], dtype=np.float32)
    p_map = np.ones(ref.shape[:3], dtype=np.float32)
    diff_map = (stack_fe - stack_rest).mean(axis=0)

    in_mask = np.argwhere(mask_data)
    log.info(f"  Running paired t-test (rest vs. fe) across {len(reho_rest)} subjects …")

    for x, y, z in in_mask:
        a = stack_rest[:, x, y, z]
        b = stack_fe[:, x, y, z]
        if (a - b).std() < 1e-10:
            continue
        t, p = stats.ttest_rel(b, a)   # fe > rest = positive t
        t_map[x, y, z] = t
        p_map[x, y, z] = p

    t_img = nib.Nifti1Image(t_map, ref.affine)
    p_img = nib.Nifti1Image(p_map, ref.affine)
    diff_img = nib.Nifti1Image(diff_map, ref.affine)
    return t_img, p_img, diff_img


def threshold_t_map(t_img: nib.Nifti1Image, p_img: nib.Nifti1Image,
                    p_thr: float = P_THRESHOLD) -> nib.Nifti1Image:
    """Zero out t-values where p > threshold."""
    t_data = t_img.get_fdata(dtype=np.float32).copy()
    p_data = p_img.get_fdata(dtype=np.float32)
    t_data[p_data > p_thr] = 0.0
    return nib.Nifti1Image(t_data, t_img.affine)


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

def plot_reho_map(reho_img: nib.Nifti1Image, title: str, out_path: Path,
                  threshold: float = Z_THRESHOLD, cmap: str = "hot"):
    display = plotting.plot_stat_map(
        reho_img, display_mode="z", cut_coords=8,
        title=title, colorbar=True, cmap=cmap, threshold=threshold,
    )
    display.savefig(str(out_path), dpi=150)
    display.close()


def plot_roi_bar(roi_dict: dict, subject: str, task: str, out_path: Path,
                 top_n: int = 20):
    """Bar chart of top-N ROIs by absolute ReHo value."""
    sorted_rois = sorted(roi_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    labels, vals = zip(*sorted_rois)
    colors = ["steelblue" if v >= 0 else "salmon" for v in vals]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(labels)), vals, color=colors, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Normalized ReHo (z-score)")
    ax.set_title(f"{subject} task-{task} | Top {top_n} AAL ROIs")
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def plot_group_roi_comparison(rest_roi_means: dict, fe_roi_means: dict,
                               out_path: Path, top_n: int = 20):
    """Side-by-side group mean ROI ReHo for rest vs. emotion task."""
    all_rois = sorted(rest_roi_means.keys(),
                      key=lambda r: abs(rest_roi_means.get(r, 0) + fe_roi_means.get(r, 0)),
                      reverse=True)[:top_n]

    rest_vals = [rest_roi_means.get(r, 0) for r in all_rois]
    fe_vals = [fe_roi_means.get(r, 0) for r in all_rois]
    x = np.arange(len(all_rois))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 0.2, rest_vals, 0.4, label="Rest", color="steelblue", alpha=0.85)
    ax.bar(x + 0.2, fe_vals, 0.4, label="Emotion Task", color="darkorange", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(all_rois, rotation=90, fontsize=6)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Mean Normalized ReHo (z-score)")
    ax.set_title(f"Group Mean ReHo by ROI | Rest vs. Emotion Task (Top {top_n})")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Per-subject pipeline
# ═══════════════════════════════════════════════════════════════════════════

def process_subject(subject: str, tasks: list, mask_img: nib.Nifti1Image):
    sub_out = REHO_DIR / subject
    sub_out.mkdir(exist_ok=True)
    sub_fig = FIG_DIR / subject
    sub_fig.mkdir(exist_ok=True)

    mask_data = mask_img.get_fdata().astype(bool)
    results = {}

    for task in tasks:
        bold_path = PREPROC_DIR / subject / f"{subject}_task-{task}_bold_preproc.nii.gz"
        if not bold_path.exists():
            log.warning(f"  {bold_path.name} not found — skipping")
            continue

        log.info(f"\n{'='*60}\n  ReHo: {subject}  task-{task}\n{'='*60}")
        img = nib.load(str(bold_path))

        # Resample mask to match functional if needed
        mask_res = image.resample_to_img(mask_img, image.index_img(img, 0),
                                          interpolation="nearest")
        mask_res_data = mask_res.get_fdata().astype(bool)

        # 1. Compute ReHo
        reho_img = compute_reho(img, mask_res_data)
        raw_path = sub_out / f"{subject}_task-{task}_reho_raw.nii.gz"
        nib.save(reho_img, str(raw_path))

        # 2. Normalize
        z_reho_img = normalize_reho(reho_img, mask_res)
        z_path = sub_out / f"{subject}_task-{task}_reho_z.nii.gz"
        nib.save(z_reho_img, str(z_path))
        log.info(f"  Saved normalized ReHo: {z_path.name}")

        # 3. ROI extraction
        try:
            roi_vals = extract_roi_values(z_reho_img)
            roi_path = sub_out / f"{subject}_task-{task}_reho_roi.json"
            with open(roi_path, "w") as f:
                json.dump({k: float(v) for k, v in roi_vals.items()}, f, indent=2)
            plot_roi_bar(roi_vals, subject, task,
                         sub_fig / f"{subject}_task-{task}_reho_roi.png")
        except Exception as e:
            log.warning(f"  ROI extraction failed: {e}")
            roi_vals = {}

        # 4. Individual ReHo map plot
        reho_thresh = float(np.percentile(
            z_reho_img.get_fdata()[mask_res_data], 85))
        plot_reho_map(z_reho_img,
                      f"{subject} task-{task} ReHo (z)",
                      sub_fig / f"{subject}_task-{task}_reho_map.png",
                      threshold=max(reho_thresh, 0.5))

        results[task] = {"z_img": z_reho_img, "roi_vals": roi_vals}
        log.info(f"  Done: {subject} task-{task}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Group-level analysis
# ═══════════════════════════════════════════════════════════════════════════

def run_group_analysis(subjects: list, tasks: list, mask_img: nib.Nifti1Image):
    log.info(f"\n{'='*60}\n  Group-level ReHo analysis\n{'='*60}")

    task_imgs = {task: [] for task in tasks}
    task_roi_dicts = {task: [] for task in tasks}
    valid_subjects = {task: [] for task in tasks}

    for sub in subjects:
        for task in tasks:
            z_path = REHO_DIR / sub / f"{sub}_task-{task}_reho_z.nii.gz"
            roi_path = REHO_DIR / sub / f"{sub}_task-{task}_reho_roi.json"
            if z_path.exists():
                task_imgs[task].append(nib.load(str(z_path)))
                valid_subjects[task].append(sub)
            if roi_path.exists():
                with open(roi_path) as f:
                    task_roi_dicts[task].append(json.load(f))

    for task in tasks:
        imgs = task_imgs[task]
        if len(imgs) < 2:
            log.warning(f"  Not enough subjects for group analysis (task-{task})")
            continue

        log.info(f"  Group t-test: task-{task}  N={len(imgs)}")
        t_img, p_img, mean_img = group_one_sample_t(imgs, mask_img, valid_subjects[task])
        thr_img = threshold_t_map(t_img, p_img)

        nib.save(t_img, str(REHO_DIR / f"group_task-{task}_reho_t.nii.gz"))
        nib.save(thr_img, str(REHO_DIR / f"group_task-{task}_reho_t_thr.nii.gz"))
        nib.save(mean_img, str(REHO_DIR / f"group_task-{task}_reho_mean.nii.gz"))

        plot_reho_map(thr_img,
                      f"Group ReHo t-map | task-{task} | N={len(imgs)} (p<{P_THRESHOLD})",
                      FIG_DIR / f"group_task-{task}_reho_tmap.png",
                      threshold=2.0, cmap="cold_hot")
        plot_reho_map(mean_img,
                      f"Group mean ReHo | task-{task} | N={len(imgs)}",
                      FIG_DIR / f"group_task-{task}_reho_mean.png",
                      threshold=0.3, cmap="hot")
        log.info(f"  Group maps saved for task-{task}")

        # Group mean ROI values
        if task_roi_dicts[task]:
            all_rois = task_roi_dicts[task][0].keys()
            group_roi_mean = {
                roi: float(np.mean([d[roi] for d in task_roi_dicts[task] if roi in d]))
                for roi in all_rois
            }
            with open(REHO_DIR / f"group_task-{task}_reho_roi_means.json", "w") as f:
                json.dump(group_roi_mean, f, indent=2)

    # Paired t-test: rest vs. fe
    rest_imgs = task_imgs.get("rest", [])
    fe_imgs = task_imgs.get("fe", [])
    paired_n = min(len(rest_imgs), len(fe_imgs))
    if paired_n >= 2:
        log.info(f"  Paired t-test rest vs. fe  N={paired_n}")
        t_img, p_img, diff_img = group_paired_t(
            rest_imgs[:paired_n], fe_imgs[:paired_n], mask_img)
        thr_img = threshold_t_map(t_img, p_img)

        nib.save(thr_img, str(REHO_DIR / "group_rest-vs-fe_reho_t_thr.nii.gz"))
        nib.save(diff_img, str(REHO_DIR / "group_rest-vs-fe_reho_diff.nii.gz"))

        plot_reho_map(thr_img,
                      f"ReHo: Emotion > Rest | N={paired_n} (p<{P_THRESHOLD})",
                      FIG_DIR / "group_rest-vs-fe_reho_tmap.png",
                      threshold=2.0, cmap="cold_hot")
        log.info("  Rest vs. fe contrast saved")

    # Group ROI comparison plot
    rest_roi = {}
    fe_roi = {}
    rest_roi_path = REHO_DIR / "group_task-rest_reho_roi_means.json"
    fe_roi_path = REHO_DIR / "group_task-fe_reho_roi_means.json"
    if rest_roi_path.exists() and fe_roi_path.exists():
        with open(rest_roi_path) as f:
            rest_roi = json.load(f)
        with open(fe_roi_path) as f:
            fe_roi = json.load(f)
        plot_group_roi_comparison(rest_roi, fe_roi,
                                   FIG_DIR / "group_reho_roi_rest_vs_fe.png")
        log.info("  Group ROI comparison plot saved")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ReHo analysis for NeuroEmo dataset")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Subject IDs. Default: all in preprocessed/")
    parser.add_argument("--tasks", nargs="+", default=["rest", "fe"])
    parser.add_argument("--skip-subjects", action="store_true",
                        help="Skip per-subject computation; run group analysis only")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel jobs (requires joblib)")
    args = parser.parse_args()

    if args.subjects:
        subjects = args.subjects
    else:
        subjects = sorted([d.name for d in PREPROC_DIR.iterdir()
                           if d.is_dir() and d.name.startswith("sub-")])

    log.info(f"ReHo analysis: {len(subjects)} subjects  tasks={args.tasks}")

    # Shared MNI brain mask
    mask_img = load_mni152_brain_mask(resolution=3)

    if not args.skip_subjects:
        if args.n_jobs > 1:
            from joblib import Parallel, delayed
            Parallel(n_jobs=args.n_jobs)(
                delayed(process_subject)(sub, args.tasks, mask_img)
                for sub in subjects
            )
        else:
            for sub in subjects:
                try:
                    process_subject(sub, args.tasks, mask_img)
                except Exception as e:
                    log.error(f"FAILED {sub}: {e}", exc_info=True)

    run_group_analysis(subjects, args.tasks, mask_img)
    log.info("\nReHo analysis complete.")


if __name__ == "__main__":
    main()
