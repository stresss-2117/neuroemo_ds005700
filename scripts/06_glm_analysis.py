"""
GLM-based Emotion Activation Analysis for NeuroEmo ds005700
============================================================
Uses known stimulus timing to identify which brain regions activate
for each of the 5 emotions: Calm, Afraid, Delighted, Depressed, Excited.

Stimulus design (same for all subjects, all runs):
  Onset(s)  Duration(s)  Condition
  0         30           calm
  30        30           white_noise
  60        30           afraid
  90        30           white_noise
  120       30           delighted
  150       30           white_noise
  180       30           depressed
  210       30           white_noise
  240       30           excited
  270       30           white_noise
  300       30           delighted
  330       30           white_noise
  360       30           depressed
  390       30           white_noise
  420       30           calm
  450       30           white_noise
  480       30           excited
  510       30           white_noise
  540       30           afraid
  570       30           white_noise

Pipeline:
  1. Build events DataFrame from known timing
  2. First-level GLM per subject (HRF convolution + motion confounds)
  3. Compute 5 emotion contrasts (each emotion > white_noise baseline)
  4. Compute emotion comparison contrasts (e.g. afraid > calm)
  5. Second-level group analysis (one-sample t-test across subjects)
  6. FDR correction + visualization

Reference:
  Friston et al. (1994) Statistical parametric maps in functional imaging.
  Human Brain Mapping, 2, 189-210.

Usage:
  python 06_glm_analysis.py [--subjects sub-01 ...] [--n-jobs 1]
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
import pandas as pd
from nilearn import image, plotting
from nilearn.datasets import load_mni152_brain_mask
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
PREPROC_DIR = BASE_DIR / "preprocessed"
GLM_DIR = BASE_DIR / "glm_analysis"
FIG_DIR = BASE_DIR / "outputs" / "figures" / "glm"
LOG_DIR = BASE_DIR / "outputs" / "logs"

for d in [GLM_DIR, FIG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────
_sh = logging.StreamHandler(sys.stdout)
_sh.stream = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[_sh, logging.FileHandler(LOG_DIR / "06_glm.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

# ── Stimulus timing (same for all subjects) ────────────────────────────────
EVENTS_TABLE = [
    #  onset   duration   condition
    (0,        30,        "calm"),
    (30,       30,        "white_noise"),
    (60,       30,        "afraid"),
    (90,       30,        "white_noise"),
    (120,      30,        "delighted"),
    (150,      30,        "white_noise"),
    (180,      30,        "depressed"),
    (210,      30,        "white_noise"),
    (240,      30,        "excited"),
    (270,      30,        "white_noise"),
    (300,      30,        "delighted"),
    (330,      30,        "white_noise"),
    (360,      30,        "depressed"),
    (390,      30,        "white_noise"),
    (420,      30,        "calm"),
    (450,      30,        "white_noise"),
    (480,      30,        "excited"),
    (510,      30,        "white_noise"),
    (540,      30,        "afraid"),
    (570,      30,        "white_noise"),
]

EMOTIONS = ["calm", "afraid", "delighted", "depressed", "excited"]

# Contrasts: each emotion vs white noise baseline
EMOTION_CONTRASTS = {emo: emo for emo in EMOTIONS}

# Pairwise comparisons of interest
PAIRWISE_CONTRASTS = {
    "afraid_vs_calm"      : "afraid - calm",
    "excited_vs_depressed": "excited - depressed",
    "delighted_vs_depressed": "delighted - depressed",
    "afraid_vs_depressed" : "afraid - depressed",
    "excited_vs_calm"     : "excited - calm",
}

# Display parameters
T_THRESHOLD = 3.0      # minimum |t| for display
FDR_ALPHA = 0.05


# ═══════════════════════════════════════════════════════════════════════════
# Events and confounds
# ═══════════════════════════════════════════════════════════════════════════

def make_events_df(n_vols: int, TR: float) -> pd.DataFrame:
    """
    Build events DataFrame trimmed to actual scan duration.
    White noise blocks serve as the implicit baseline in each contrast.
    """
    scan_duration = n_vols * TR
    rows = []
    for onset, duration, condition in EVENTS_TABLE:
        if onset >= scan_duration:
            break                          # event starts after scan ends
        # Clip duration if event extends past end of scan
        clipped_dur = min(duration, scan_duration - onset)
        rows.append({"onset": float(onset),
                     "duration": float(clipped_dur),
                     "trial_type": condition})
    df = pd.DataFrame(rows)
    log.info(f"  Events: {len(df)} blocks within {scan_duration:.1f}s scan")
    return df


def load_motion_confounds(subject: str, task: str,
                          n_vols: int) -> pd.DataFrame | None:
    """Load 6 motion parameters + 6 derivatives as confound regressors."""
    mp_path = PREPROC_DIR / subject / f"{subject}_task-{task}_motion.txt"
    if not mp_path.exists():
        return None
    mp = np.loadtxt(str(mp_path))[:n_vols]   # trim to actual volumes
    deriv = np.vstack([mp[:1], np.diff(mp, axis=0)])
    confounds = np.hstack([mp, deriv])
    cols = ["tx","ty","tz","rx","ry","rz",
            "dtx","dty","dtz","drx","dry","drz"]
    return pd.DataFrame(confounds, columns=cols)


def get_tr(subject: str, task: str) -> float:
    json_path = BASE_DIR / "raw" / subject / "func" / f"{subject}_task-{task}_bold.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f).get("RepetitionTime", 2.0)
    return 2.027


# ═══════════════════════════════════════════════════════════════════════════
# First-level GLM (per subject)
# ═══════════════════════════════════════════════════════════════════════════

def run_first_level(subject: str) -> dict:
    """
    Fit GLM for task-fe and compute z-maps for each emotion contrast.

    The GLM:
    - Models each emotion block as a boxcar convolved with the canonical HRF
    - White noise blocks are NOT explicitly modeled → they become the implicit baseline
    - Includes 12 motion confound regressors
    - Uses AR(1) noise model to handle temporal autocorrelation

    Returns dict: {contrast_name: z_score_img}
    """
    bold_path = PREPROC_DIR / subject / f"{subject}_task-fe_bold_preproc.nii.gz"
    if not bold_path.exists():
        log.warning(f"  {subject}: preprocessed file not found")
        return {}

    img = nib.load(str(bold_path))
    n_vols = img.shape[3]
    TR = get_tr(subject, "fe")

    log.info(f"\n{'='*55}\n  GLM: {subject}  n_vols={n_vols}  TR={TR:.4f}s\n{'='*55}")

    events_df = make_events_df(n_vols, TR)
    confounds_df = load_motion_confounds(subject, "fe", n_vols)

    # ── Fit GLM ────────────────────────────────────────────────────────
    glm = FirstLevelModel(
        t_r=TR,
        hrf_model="spm",          # canonical SPM double-gamma HRF
        drift_model="cosine",     # low-frequency drift removal
        high_pass=0.01,           # matches preprocessing filter
        noise_model="ar1",        # correct for temporal autocorrelation
        standardize=False,
        mask_img=load_mni152_brain_mask(resolution=3),
        minimize_memory=True,
        verbose=0,
    )
    glm.fit(img, events=events_df, confounds=confounds_df)
    log.info("  GLM fitted")

    # ── Compute contrasts ───────────────────────────────────────────────
    sub_out = GLM_DIR / subject
    sub_out.mkdir(exist_ok=True)
    results = {}

    # 1. Each emotion vs implicit baseline (white noise)
    for emo in EMOTIONS:
        z_img = glm.compute_contrast(emo, output_type="z_score")
        out_path = sub_out / f"{subject}_{emo}_vs_noise_z.nii.gz"
        nib.save(z_img, str(out_path))
        results[emo] = z_img
        peak_z = float(z_img.get_fdata()[
            load_mni152_brain_mask(resolution=3).get_fdata().astype(bool)].max())
        log.info(f"  {emo} > noise: peak z={peak_z:.2f}")

    # 2. Pairwise emotion comparisons
    for name, formula in PAIRWISE_CONTRASTS.items():
        try:
            z_img = glm.compute_contrast(formula, output_type="z_score")
            out_path = sub_out / f"{subject}_{name}_z.nii.gz"
            nib.save(z_img, str(out_path))
            results[name] = z_img
            log.info(f"  {name}: saved")
        except Exception as e:
            log.warning(f"  {name} failed: {e}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Second-level GLM (group analysis)
# ═══════════════════════════════════════════════════════════════════════════

def run_second_level(subjects: list, contrast_name: str) -> dict:
    """
    One-sample t-test across subjects' first-level z-maps.
    Tests H0: mean activation = 0 across the group.
    Returns FDR-thresholded z-map.
    """
    z_imgs = []
    for sub in subjects:
        p = GLM_DIR / sub / f"{sub}_{contrast_name}_vs_noise_z.nii.gz" \
            if "_vs_" not in contrast_name else \
            GLM_DIR / sub / f"{sub}_{contrast_name}_z.nii.gz"
        # Handle naming
        p1 = GLM_DIR / sub / f"{sub}_{contrast_name}_vs_noise_z.nii.gz"
        p2 = GLM_DIR / sub / f"{sub}_{contrast_name}_z.nii.gz"
        path = p1 if p1.exists() else p2 if p2.exists() else None
        if path:
            z_imgs.append(nib.load(str(path)))

    if len(z_imgs) < 2:
        log.warning(f"  Not enough subjects for group analysis: {contrast_name}")
        return {}

    log.info(f"  Second-level: {contrast_name}  N={len(z_imgs)}")

    # Design matrix: intercept only (one-sample t-test)
    design_matrix = pd.DataFrame({"intercept": np.ones(len(z_imgs))})

    second_level = SecondLevelModel(smoothing_fwhm=None)
    second_level.fit(z_imgs, design_matrix=design_matrix)

    # Group z-map
    group_z = second_level.compute_contrast(
        second_level_stat_type="t", output_type="z_score")

    # FDR threshold
    thresholded, threshold = threshold_stats_img(
        group_z, alpha=FDR_ALPHA, height_control="fdr", cluster_threshold=0)

    return {
        "group_z"    : group_z,
        "thresholded": thresholded,
        "threshold"  : threshold,
        "n"          : len(z_imgs),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

# Color scheme per emotion
EMOTION_COLORS = {
    "calm"      : "Blues",
    "afraid"    : "Reds",
    "delighted" : "YlOrRd",
    "depressed" : "PuBu",
    "excited"   : "OrRd",
}

def plot_group_emotion_map(result: dict, emotion: str, out_dir: Path):
    """Brain map for one emotion vs white noise baseline."""
    if not result:
        return
    thr_img   = result["thresholded"]
    threshold = result["threshold"]
    n         = result["n"]
    cmap      = EMOTION_COLORS.get(emotion, "hot")

    # Axial slices
    display = plotting.plot_stat_map(
        thr_img, display_mode="z", cut_coords=8,
        title=f"{emotion.upper()} > Baseline | N={n} | FDR q<{FDR_ALPHA}",
        colorbar=True, cmap=cmap, threshold=max(threshold, 2.0),
    )
    display.savefig(str(out_dir / f"group_{emotion}_axial.png"), dpi=150)
    display.close()

    # Glass brain
    display = plotting.plot_glass_brain(
        thr_img, display_mode="lyrz",
        title=f"{emotion.upper()} > Baseline (glass brain, N={n})",
        colorbar=True, cmap=cmap, threshold=max(threshold, 2.0),
        plot_abs=False,
    )
    display.savefig(str(out_dir / f"group_{emotion}_glass.png"), dpi=150)
    display.close()
    log.info(f"  Maps saved: {emotion}")


def plot_all_emotions_summary(results: dict, out_dir: Path):
    """
    Single figure with all 5 emotions side by side (one row per emotion,
    showing a representative axial slice at z=0 to z=30).
    """
    emotions = [e for e in EMOTIONS if e in results and results[e]]
    if not emotions:
        return

    fig, axes = plt.subplots(len(emotions), 1,
                             figsize=(16, 3 * len(emotions)))
    if len(emotions) == 1:
        axes = [axes]

    for ax, emo in zip(axes, emotions):
        thr_img = results[emo]["thresholded"]
        threshold = max(results[emo]["threshold"], 2.0)
        n = results[emo]["n"]
        cmap = EMOTION_COLORS.get(emo, "hot")

        display = plotting.plot_stat_map(
            thr_img, display_mode="z", cut_coords=6,
            axes=ax, colorbar=True, cmap=cmap,
            threshold=threshold,
            title=f"{emo.upper()} > Baseline  (N={n}, FDR q<{FDR_ALPHA})",
        )

    plt.tight_layout()
    out_path = out_dir / "all_emotions_summary.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    log.info("  All-emotions summary figure saved")


def plot_pairwise_contrast(result: dict, name: str, out_dir: Path):
    """Brain map for a pairwise emotion contrast."""
    if not result:
        return
    display = plotting.plot_glass_brain(
        result["thresholded"], display_mode="lyrz",
        title=f"{name.replace('_', ' ').title()} | N={result['n']} FDR q<{FDR_ALPHA}",
        colorbar=True, cmap="cold_hot",
        threshold=max(result["threshold"], 2.0), plot_abs=False,
    )
    display.savefig(str(out_dir / f"group_{name}_glass.png"), dpi=150)
    display.close()


def plot_design_matrix(subject: str, events_df: pd.DataFrame,
                       TR: float, n_vols: int, out_dir: Path):
    """Visualize the GLM design matrix for QC."""
    from nilearn.plotting import plot_design_matrix
    from nilearn.glm.first_level import make_first_level_design_matrix

    frame_times = np.arange(n_vols) * TR
    dm = make_first_level_design_matrix(
        frame_times, events_df, hrf_model="spm",
        drift_model="cosine", high_pass=0.01)

    fig, ax = plt.subplots(figsize=(14, 4))
    plot_design_matrix(dm, axes=ax)
    ax.set_title(f"GLM Design Matrix | {subject} task-fe")
    plt.tight_layout()
    fig.savefig(str(out_dir / f"{subject}_design_matrix.png"), dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GLM emotion activation analysis for NeuroEmo dataset")
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--skip-first-level", action="store_true",
                        help="Skip per-subject GLM; run group analysis only")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    if args.subjects:
        subjects = args.subjects
    else:
        subjects = sorted([d.name for d in PREPROC_DIR.iterdir()
                           if d.is_dir() and d.name.startswith("sub-")])

    log.info(f"GLM analysis: {len(subjects)} subjects")

    # ── First-level (per subject) ──────────────────────────────────────
    if not args.skip_first_level:
        # Plot design matrix once for QC
        sample_sub = subjects[0]
        sample_img = nib.load(str(
            PREPROC_DIR / sample_sub / f"{sample_sub}_task-fe_bold_preproc.nii.gz"))
        sample_tr = get_tr(sample_sub, "fe")
        sample_n = sample_img.shape[3]
        plot_design_matrix(sample_sub, make_events_df(sample_n, sample_tr),
                           sample_tr, sample_n, FIG_DIR)
        log.info("  Design matrix plot saved")

        if args.n_jobs > 1:
            from joblib import Parallel, delayed
            Parallel(n_jobs=args.n_jobs)(
                delayed(run_first_level)(sub) for sub in subjects)
        else:
            for sub in subjects:
                try:
                    run_first_level(sub)
                except Exception as e:
                    log.error(f"  FAILED {sub}: {e}", exc_info=True)

    # ── Second-level (group) ───────────────────────────────────────────
    log.info(f"\n{'='*55}\n  Group-level GLM analysis\n{'='*55}")

    # Collect subjects with completed first-level
    valid_subjects = [
        sub for sub in subjects
        if any((GLM_DIR / sub / f"{sub}_{emo}_vs_noise_z.nii.gz").exists()
               for emo in EMOTIONS)
    ]
    log.info(f"  Valid subjects for group analysis: {len(valid_subjects)}")

    # Emotion vs baseline contrasts
    group_results = {}
    for emo in EMOTIONS:
        result = run_second_level(valid_subjects, emo)
        group_results[emo] = result
        if result:
            nib.save(result["thresholded"],
                     str(GLM_DIR / f"group_{emo}_vs_noise_z_fdr.nii.gz"))
            nib.save(result["group_z"],
                     str(GLM_DIR / f"group_{emo}_vs_noise_z.nii.gz"))
            plot_group_emotion_map(result, emo, FIG_DIR)

    # All-emotions summary figure
    plot_all_emotions_summary(group_results, FIG_DIR)

    # Pairwise contrasts
    log.info("\n  Pairwise contrasts ...")
    for name in PAIRWISE_CONTRASTS:
        result = run_second_level(valid_subjects, name)
        if result:
            nib.save(result["thresholded"],
                     str(GLM_DIR / f"group_{name}_z_fdr.nii.gz"))
            plot_pairwise_contrast(result, name, FIG_DIR)

    # Save summary
    summary = {
        "n_subjects": len(valid_subjects),
        "subjects": valid_subjects,
        "emotions": EMOTIONS,
        "fdr_alpha": FDR_ALPHA,
        "contrasts": {
            emo: {
                "threshold": group_results[emo].get("threshold"),
                "n": group_results[emo].get("n"),
            }
            for emo in EMOTIONS if group_results.get(emo)
        }
    }
    with open(GLM_DIR / "glm_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("\nGLM analysis complete.")
    log.info(f"Figures: {FIG_DIR}")
    log.info(f"Stats:   {GLM_DIR}")


if __name__ == "__main__":
    main()
