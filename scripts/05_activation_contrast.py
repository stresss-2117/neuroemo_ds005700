"""
Activation Contrast: Emotion Task (task-fe) vs Resting State (task-rest)
=========================================================================
Since no stimulus event files exist, we use fALFF (fractional Amplitude of
Low-Frequency Fluctuations) as the measure of regional brain activity.

fALFF = power in 0.01-0.1 Hz / total power across 0-0.25 Hz
      → captures spontaneous BOLD fluctuation strength per voxel
      → higher fALFF = more active / more fluctuating region

Pipeline:
  1. Compute fALFF map per subject per task
  2. Paired t-test: fe > rest  and  rest > fe  (two-tailed)
  3. FDR correction across voxels
  4. ROI-level fALFF extraction (Harvard-Oxford atlas)
  5. Visualizations: whole-brain contrast map, ROI bar chart, glass brain

Reference:
  Zou et al. (2008) "An improved approach to detection of amplitude of
  low-frequency fluctuation (ALFF) for resting-state fMRI"
  Journal of Neuroscience Methods, 172(1), 137-141.

Usage:
  python 05_activation_contrast.py [--subjects sub-01 ...] [--n-jobs 1]
"""

import argparse
import json
import logging
import ssl
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
from scipy.signal import welch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
PREPROC_DIR = BASE_DIR / "preprocessed"
ACT_DIR = BASE_DIR / "activation_contrast"
FIG_DIR = BASE_DIR / "outputs" / "figures" / "activation"
LOG_DIR = BASE_DIR / "outputs" / "logs"

for d in [ACT_DIR, FIG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────
_sh = logging.StreamHandler(sys.stdout)
_sh.stream = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[_sh, logging.FileHandler(LOG_DIR / "05_activation.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

# ── Parameters ─────────────────────────────────────────────────────────────
FALFF_LOW  = 0.01   # Hz  low-frequency band lower bound
FALFF_HIGH = 0.10   # Hz  low-frequency band upper bound
FDR_ALPHA  = 0.05   # FDR correction threshold
T_DISPLAY  = 2.0    # minimum |t| to show on brain maps


# ═══════════════════════════════════════════════════════════════════════════
# fALFF computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_falff(img: nib.Nifti1Image, TR: float,
                  mask_data: np.ndarray) -> nib.Nifti1Image:
    """
    Compute fractional ALFF (fALFF) for every in-mask voxel.

    fALFF(v) = sqrt(sum PSD in [0.01, 0.10] Hz)
               / sqrt(sum PSD in [0, Nyquist] Hz)

    Uses Welch's method for PSD estimation.
    """
    data = img.get_fdata(dtype=np.float32)   # (X, Y, Z, T)
    X, Y, Z, T = data.shape
    fs = 1.0 / TR
    nyquist = fs / 2.0

    # Welch parameters — nperseg ~ 1/2 of time series
    nperseg = min(T // 2, 64)

    # Compute PSD for all in-mask voxels at once (reshape → 2D)
    flat = data.reshape(-1, T)               # (N_vox, T)
    flat_mask = mask_data.flatten().astype(bool)
    in_mask_idx = np.where(flat_mask)[0]

    falff_flat = np.zeros(X * Y * Z, dtype=np.float32)

    log.info(f"  Computing fALFF for {len(in_mask_idx)} voxels …")
    freqs, _ = welch(flat[in_mask_idx[0]], fs=fs, nperseg=nperseg)
    lf_mask = (freqs >= FALFF_LOW) & (freqs <= FALFF_HIGH)

    # Vectorize: compute PSD for all in-mask voxels at once
    # welch doesn't directly support 2D — use numpy FFT instead (faster)
    chunk = flat[in_mask_idx]                # (n_in_mask, T)

    # Zero-mean each voxel
    chunk = chunk - chunk.mean(axis=1, keepdims=True)

    # FFT-based power spectrum
    fft_vals = np.fft.rfft(chunk, axis=1)    # (n_in_mask, T//2+1)
    power = (np.abs(fft_vals) ** 2) / T
    fft_freqs = np.fft.rfftfreq(T, d=TR)

    lf_idx = (fft_freqs >= FALFF_LOW) & (fft_freqs <= FALFF_HIGH)
    tot_idx = fft_freqs >= 0   # all frequencies

    alff  = np.sqrt(power[:, lf_idx].sum(axis=1))
    total = np.sqrt(power[:, tot_idx].sum(axis=1))

    with np.errstate(divide="ignore", invalid="ignore"):
        falff_vals = np.where(total > 0, alff / total, 0.0)

    falff_flat[in_mask_idx] = falff_vals.astype(np.float32)
    falff_map = falff_flat.reshape(X, Y, Z)

    log.info(f"  fALFF range: [{falff_vals.min():.4f}, {falff_vals.max():.4f}]  "
             f"mean={falff_vals.mean():.4f}")
    return nib.Nifti1Image(falff_map, img.affine, img.header)


def normalize_falff(falff_img: nib.Nifti1Image,
                    mask_data: np.ndarray) -> nib.Nifti1Image:
    """Z-score fALFF within brain mask to remove global session effects."""
    data = falff_img.get_fdata(dtype=np.float32)
    mask = mask_data.astype(bool)
    brain_vals = data[mask]
    z_data = np.zeros_like(data)
    z_data[mask] = (brain_vals - brain_vals.mean()) / (brain_vals.std() + 1e-12)
    return nib.Nifti1Image(z_data, falff_img.affine, falff_img.header)


# ═══════════════════════════════════════════════════════════════════════════
# Atlas loading (with SSL fallback — same pattern as ReHo script)
# ═══════════════════════════════════════════════════════════════════════════

def load_atlas():
    orig = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        try:
            from nilearn.datasets import fetch_atlas_harvard_oxford
            atlas = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
            return atlas.maps, atlas.labels
        except Exception:
            pass
        try:
            from nilearn.datasets import fetch_atlas_aal
            atlas = fetch_atlas_aal(version="SPM12")
            return atlas.maps, atlas.labels
        except Exception:
            pass
        raise RuntimeError("Atlas download failed")
    finally:
        ssl._create_default_https_context = orig


# ═══════════════════════════════════════════════════════════════════════════
# Group statistics
# ═══════════════════════════════════════════════════════════════════════════

def fdr_correct(p_map: np.ndarray, mask: np.ndarray,
                alpha: float = FDR_ALPHA) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction applied only within brain mask.
    Returns boolean map of significant voxels.
    """
    p_vals = p_map[mask]
    n = len(p_vals)
    sorted_idx = np.argsort(p_vals)
    sorted_p = p_vals[sorted_idx]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    below = sorted_p <= thresholds
    if not below.any():
        return np.zeros_like(mask, dtype=bool)
    max_k = np.where(below)[0].max()
    cutoff = sorted_p[max_k]

    sig_flat = p_vals <= cutoff
    sig_map = np.zeros_like(mask, dtype=bool)
    sig_map[mask] = sig_flat
    return sig_map


def paired_contrast(fe_imgs: list, rest_imgs: list,
                    mask_img: nib.Nifti1Image) -> dict:
    """
    Voxel-wise paired t-test: task-fe fALFF vs task-rest fALFF.
    Returns t-map, p-map, mean difference map, FDR-thresholded t-map.
    """
    ref = fe_imgs[0]
    mask_data = mask_img.get_fdata().astype(bool)

    log.info(f"  Stacking images for {len(fe_imgs)} subjects …")
    resample = lambda imgs: np.stack(
        [image.resample_to_img(img, ref, interpolation="continuous")
         .get_fdata(dtype=np.float32) for img in imgs], axis=0)

    fe_stack   = resample(fe_imgs)      # (N, X, Y, Z)
    rest_stack = resample(rest_imgs)

    diff = fe_stack - rest_stack        # positive = fe > rest

    t_map    = np.zeros(ref.shape[:3], dtype=np.float32)
    p_map    = np.ones(ref.shape[:3],  dtype=np.float32)
    mean_map = diff.mean(axis=0)

    in_mask = np.argwhere(mask_data)
    log.info(f"  Running paired t-test across {len(fe_imgs)} subject pairs …")

    # Vectorized t-test over all in-mask voxels
    diff_masked = diff[:, mask_data]            # (N, n_vox)
    n = diff_masked.shape[0]
    mean_d = diff_masked.mean(axis=0)
    std_d  = diff_masked.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_vals = np.where(std_d > 1e-10, mean_d / (std_d / np.sqrt(n)), 0.0)
    p_vals = 2 * stats.t.sf(np.abs(t_vals), df=n - 1)  # two-tailed

    t_map[mask_data] = t_vals.astype(np.float32)
    p_map[mask_data] = p_vals.astype(np.float32)

    # FDR correction
    sig_mask = fdr_correct(p_map, mask_data)
    t_fdr = t_map.copy()
    t_fdr[~sig_mask] = 0.0

    log.info(f"  Significant voxels (FDR q<{FDR_ALPHA}): {sig_mask.sum()}")

    return {
        "t_img"      : nib.Nifti1Image(t_map,  ref.affine),
        "p_img"      : nib.Nifti1Image(p_map,  ref.affine),
        "t_fdr_img"  : nib.Nifti1Image(t_fdr,  ref.affine),
        "mean_diff"  : nib.Nifti1Image(mean_map, ref.affine),
        "n_sig_vox"  : int(sig_mask.sum()),
        "n_fe_gt_rest": int((t_fdr > 0).sum()),
        "n_rest_gt_fe": int((t_fdr < 0).sum()),
    }


# ═══════════════════════════════════════════════════════════════════════════
# ROI extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_roi_contrast(fe_imgs: list, rest_imgs: list) -> dict:
    """Mean fALFF per atlas ROI for each task, across subjects."""
    try:
        maps_img, labels = load_atlas()
    except RuntimeError as e:
        log.warning(f"  Atlas unavailable: {e}")
        return {}

    masker = NiftiLabelsMasker(
        labels_img=maps_img, labels=labels,
        resampling_target="data", standardize=False)

    def mean_roi(imgs):
        vals = []
        for img in imgs:
            img4d = image.new_img_like(img, img.get_fdata()[..., np.newaxis])
            v = masker.fit_transform(img4d)
            vals.append(v[0])
        return np.array(vals)   # (N_subjects, N_rois)

    fe_roi   = mean_roi(fe_imgs)
    rest_roi = mean_roi(rest_imgs)

    # Use masker.labels_ which matches the actual extracted columns
    # (NiftiLabelsMasker drops background label, so masker.labels_ is authoritative)
    fitted_labels = masker.labels_ if hasattr(masker, "labels_") else [
        l for l in labels if l != "Background"]
    if len(fitted_labels) != fe_roi.shape[1]:
        fitted_labels = [l for l in labels if l != "Background"]
    if len(fitted_labels) != fe_roi.shape[1]:
        fitted_labels = [f"ROI_{i}" for i in range(fe_roi.shape[1])]

    # Paired t-test per ROI
    n = min(len(fe_imgs), len(rest_imgs))
    diff = fe_roi[:n] - rest_roi[:n]
    t_vals, p_vals = stats.ttest_rel(fe_roi[:n], rest_roi[:n], axis=0)

    roi_results = {}
    for i, lbl in enumerate(fitted_labels):
        roi_results[lbl] = {
            "fe_mean"   : float(fe_roi[:, i].mean()),
            "rest_mean" : float(rest_roi[:, i].mean()),
            "diff_mean" : float(diff[:, i].mean()),
            "t"         : float(t_vals[i]),
            "p"         : float(p_vals[i]),
        }
    return roi_results


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

def plot_contrast_brain(t_fdr_img: nib.Nifti1Image, n: int, out_dir: Path):
    """Whole-brain t-map: red = fe > rest, blue = rest > fe."""

    # Axial slices
    display = plotting.plot_stat_map(
        t_fdr_img,
        display_mode="z", cut_coords=10,
        title=f"Emotion Task > Rest | fALFF contrast (FDR q<{FDR_ALPHA}) | N={n}",
        colorbar=True, cmap="cold_hot", threshold=T_DISPLAY,
    )
    display.savefig(str(out_dir / "contrast_fe_vs_rest_axial.png"), dpi=150)
    display.close()

    # Glass brain — shows peaks from all angles
    display = plotting.plot_glass_brain(
        t_fdr_img, display_mode="lyrz",
        title=f"Emotion > Rest | Glass Brain (N={n})",
        colorbar=True, cmap="cold_hot", threshold=T_DISPLAY,
        plot_abs=False,
    )
    display.savefig(str(out_dir / "contrast_fe_vs_rest_glass.png"), dpi=150)
    display.close()

    log.info("  Brain contrast maps saved")


def plot_roi_contrast(roi_results: dict, out_dir: Path, top_n: int = 25):
    """Bar chart: top ROIs by |t| with significance markers."""
    if not roi_results:
        return

    # Sort by absolute t-value
    sorted_rois = sorted(roi_results.items(),
                         key=lambda x: abs(x[1]["t"]), reverse=True)[:top_n]
    labels  = [r[0] for r in sorted_rois]
    t_vals  = [r[1]["t"] for r in sorted_rois]
    p_vals  = [r[1]["p"] for r in sorted_rois]

    colors = ["#d73027" if t > 0 else "#4575b4" for t in t_vals]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(labels)), t_vals, color=colors,
                  edgecolor="white", linewidth=0.5)

    # Significance markers
    for i, p in enumerate(p_vals):
        marker = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if marker:
            y_pos = t_vals[i] + (0.1 if t_vals[i] >= 0 else -0.2)
            ax.text(i, y_pos, marker, ha="center", fontsize=8, color="black")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("t-statistic (fe − rest)")
    ax.set_title(f"ROI fALFF Contrast: Emotion Task vs Rest | Top {top_n} regions\n"
                 f"Red = more active during emotion  |  Blue = more active at rest")

    # Custom legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#d73027", label="Emotion > Rest"),
                        Patch(color="#4575b4", label="Rest > Emotion")],
              fontsize=9, loc="upper right")

    plt.tight_layout()
    fig.savefig(str(out_dir / "roi_contrast_bar.png"), dpi=150)
    plt.close(fig)
    log.info("  ROI contrast bar chart saved")


def plot_falff_comparison(fe_imgs: list, rest_imgs: list,
                           mask_img: nib.Nifti1Image, out_dir: Path):
    """Side-by-side group mean fALFF maps for each task."""
    ref = fe_imgs[0]
    resample = lambda imgs: image.mean_img(
        [image.resample_to_img(img, ref) for img in imgs])

    fe_mean   = resample(fe_imgs)
    rest_mean = resample(rest_imgs)

    for task, mean_img in [("fe", fe_mean), ("rest", rest_mean)]:
        display = plotting.plot_stat_map(
            mean_img, display_mode="z", cut_coords=8,
            title=f"Group mean fALFF | task-{task} (N={len(fe_imgs)})",
            colorbar=True, cmap="hot", threshold=0.1,
        )
        display.savefig(str(out_dir / f"group_falff_task-{task}.png"), dpi=150)
        display.close()

    log.info("  Group mean fALFF maps saved")


def plot_subject_summary(subjects: list, fe_falff_means: list,
                          rest_falff_means: list, out_dir: Path):
    """Scatter: per-subject mean fALFF for fe vs rest."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(rest_falff_means, fe_falff_means, color="steelblue", s=60, zorder=3)

    lim = [min(rest_falff_means + fe_falff_means) * 0.95,
           max(rest_falff_means + fe_falff_means) * 1.05]
    ax.plot(lim, lim, "k--", linewidth=0.8, label="Identity (fe = rest)")

    for i, sub in enumerate(subjects):
        ax.annotate(sub.replace("sub-", ""),
                    (rest_falff_means[i], fe_falff_means[i]),
                    fontsize=7, ha="center", va="bottom")

    ax.set_xlabel("Mean whole-brain fALFF | task-rest")
    ax.set_ylabel("Mean whole-brain fALFF | task-fe")
    ax.set_title("Per-subject whole-brain fALFF: Emotion vs Rest")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(str(out_dir / "subject_falff_scatter.png"), dpi=150)
    plt.close(fig)
    log.info("  Subject scatter plot saved")


# ═══════════════════════════════════════════════════════════════════════════
# Per-subject fALFF
# ═══════════════════════════════════════════════════════════════════════════

def get_tr(subject: str, task: str) -> float:
    json_path = BASE_DIR / "raw" / subject / "func" / f"{subject}_task-{task}_bold.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f).get("RepetitionTime", 2.0)
    return 2.0


def process_subject(subject: str, mask_img: nib.Nifti1Image) -> dict:
    sub_out = ACT_DIR / subject
    sub_out.mkdir(exist_ok=True)

    results = {}
    mask_data = mask_img.get_fdata().astype(bool)

    for task in ["rest", "fe"]:
        bold_path = PREPROC_DIR / subject / f"{subject}_task-{task}_bold_preproc.nii.gz"
        if not bold_path.exists():
            log.warning(f"  Missing: {bold_path.name}")
            continue

        log.info(f"  {subject} task-{task}")
        img = nib.load(str(bold_path))
        TR = get_tr(subject, task)

        # Resample mask to functional space
        mask_res = image.resample_to_img(mask_img, image.index_img(img, 0),
                                          interpolation="nearest")
        mask_res_data = mask_res.get_fdata().astype(bool)

        falff_img = compute_falff(img, TR, mask_res_data)
        z_falff   = normalize_falff(falff_img, mask_res_data)

        nib.save(falff_img, str(sub_out / f"{subject}_task-{task}_falff.nii.gz"))
        nib.save(z_falff,   str(sub_out / f"{subject}_task-{task}_falff_z.nii.gz"))

        brain_mean = float(falff_img.get_fdata()[mask_res_data].mean())
        results[task] = {"img": z_falff, "brain_mean": brain_mean}
        log.info(f"    Saved fALFF  brain_mean={brain_mean:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="fALFF-based activation contrast: emotion task vs rest")
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--skip-subjects", action="store_true",
                        help="Skip per-subject fALFF; run group analysis only")
    args = parser.parse_args()

    if args.subjects:
        subjects = args.subjects
    else:
        subjects = sorted([d.name for d in PREPROC_DIR.iterdir()
                           if d.is_dir() and d.name.startswith("sub-")])

    log.info(f"Activation contrast: {len(subjects)} subjects")

    mask_img = load_mni152_brain_mask(resolution=3)

    # ── Per-subject fALFF ──────────────────────────────────────────────────
    if not args.skip_subjects:
        if args.n_jobs > 1:
            from joblib import Parallel, delayed
            all_results = Parallel(n_jobs=args.n_jobs)(
                delayed(process_subject)(sub, mask_img) for sub in subjects)
        else:
            all_results = []
            for sub in subjects:
                try:
                    all_results.append(process_subject(sub, mask_img))
                except Exception as e:
                    log.error(f"  FAILED {sub}: {e}", exc_info=True)
                    all_results.append({})

    # ── Collect paired subjects (both tasks must exist) ────────────────────
    log.info("\nCollecting paired subjects for group analysis …")
    fe_imgs, rest_imgs, paired_subs = [], [], []
    fe_means, rest_means = [], []

    for sub in subjects:
        fe_path   = ACT_DIR / sub / f"{sub}_task-fe_falff_z.nii.gz"
        rest_path = ACT_DIR / sub / f"{sub}_task-rest_falff_z.nii.gz"
        fe_raw    = ACT_DIR / sub / f"{sub}_task-fe_falff.nii.gz"
        rest_raw  = ACT_DIR / sub / f"{sub}_task-rest_falff.nii.gz"

        if fe_path.exists() and rest_path.exists():
            fe_imgs.append(nib.load(str(fe_path)))
            rest_imgs.append(nib.load(str(rest_path)))
            paired_subs.append(sub)

            m_fe   = nib.load(str(fe_raw)).get_fdata()
            m_rest = nib.load(str(rest_raw)).get_fdata()
            mask_d = mask_img.get_fdata().astype(bool)
            # resample mask to match
            mk_fe   = image.resample_to_img(mask_img,
                          image.index_img(nib.load(str(fe_path)), 0)
                          if False else nib.load(str(fe_path)),
                          interpolation="nearest").get_fdata().astype(bool)
            fe_means.append(float(m_fe[mk_fe].mean()) if mk_fe.any() else 0)
            rest_means.append(float(m_rest[mk_fe].mean()) if mk_fe.any() else 0)

    n = len(paired_subs)
    log.info(f"Paired subjects: {n}")

    if n < 2:
        log.error("Need at least 2 paired subjects for group analysis.")
        return

    # ── Group contrast ─────────────────────────────────────────────────────
    log.info(f"\n{'='*60}\n  Group contrast (N={n})\n{'='*60}")
    contrast = paired_contrast(fe_imgs, rest_imgs, mask_img)

    nib.save(contrast["t_img"],     str(ACT_DIR / "group_fe_vs_rest_t.nii.gz"))
    nib.save(contrast["t_fdr_img"], str(ACT_DIR / "group_fe_vs_rest_t_fdr.nii.gz"))
    nib.save(contrast["mean_diff"], str(ACT_DIR / "group_fe_vs_rest_diff.nii.gz"))

    log.info(f"  Significant voxels (FDR): {contrast['n_sig_vox']}")
    log.info(f"  fe > rest: {contrast['n_fe_gt_rest']} voxels")
    log.info(f"  rest > fe: {contrast['n_rest_gt_fe']} voxels")

    # ── Visualizations ─────────────────────────────────────────────────────
    plot_contrast_brain(contrast["t_fdr_img"], n, FIG_DIR)
    plot_falff_comparison(fe_imgs, rest_imgs, mask_img, FIG_DIR)
    plot_subject_summary(paired_subs, fe_means, rest_means, FIG_DIR)

    # ── ROI contrast ───────────────────────────────────────────────────────
    log.info("\nROI-level contrast …")
    roi_results = extract_roi_contrast(fe_imgs, rest_imgs)
    if roi_results:
        with open(ACT_DIR / "group_roi_contrast.json", "w") as f:
            json.dump(roi_results, f, indent=2)
        plot_roi_contrast(roi_results, FIG_DIR)

    # ── Summary JSON ───────────────────────────────────────────────────────
    summary = {
        "n_subjects": n,
        "subjects": paired_subs,
        "n_sig_voxels_fdr": contrast["n_sig_vox"],
        "n_vox_fe_gt_rest": contrast["n_fe_gt_rest"],
        "n_vox_rest_gt_fe": contrast["n_rest_gt_fe"],
        "fdr_alpha": FDR_ALPHA,
        "falff_band_hz": [FALFF_LOW, FALFF_HIGH],
    }
    with open(ACT_DIR / "group_contrast_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("\nActivation contrast complete.")
    log.info(f"Figures: {FIG_DIR}")
    log.info(f"Stats:   {ACT_DIR}")


if __name__ == "__main__":
    main()
