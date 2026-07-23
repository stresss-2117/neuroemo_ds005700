"""
Group-level visualization for NeuroEmo QPP analysis.

Generates:
  1. Group QPP spatial map (mean across subjects)
  2. QPP strength (mean correlation) bar chart per subject
  3. QPP periodicity summary (peak frequency + period)
  4. Motion QC summary across all subjects
  5. Group-level spatial map overlaid on MNI template

Usage:
  python 03_visualize.py [--tasks rest fe] [--method swc seed]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import image, plotting
from nilearn.datasets import load_mni152_brain_mask

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
PREPROC_DIR = BASE_DIR / "preprocessed"
QPP_DIR = BASE_DIR / "qpp_analysis"
FIG_DIR = BASE_DIR / "outputs" / "figures"
LOG_DIR = BASE_DIR / "outputs" / "logs"

FIG_DIR.mkdir(parents=True, exist_ok=True)

_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.stream = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        _stream_handler,
        logging.FileHandler(LOG_DIR / "03_visualize.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_summaries(subjects: list, task: str) -> list:
    records = []
    for sub in subjects:
        p = QPP_DIR / sub / f"{sub}_task-{task}_qpp_summary.json"
        if p.exists():
            with open(p) as f:
                records.append(json.load(f))
        else:
            log.warning(f"  Missing summary: {p.name}")
    return records


def load_spatial_maps(subjects: list, task: str, method: str) -> list:
    imgs = []
    for sub in subjects:
        if method == "swc":
            p = QPP_DIR / sub / f"{sub}_task-{task}_swc_qpp_template.nii.gz"
        else:
            p = QPP_DIR / sub / f"{sub}_task-{task}_seed_corr_map.nii.gz"
        if p.exists():
            imgs.append(nib.load(str(p)))
        else:
            log.warning(f"  Missing map: {p.name}")
    return imgs


def load_motion_fd(subjects: list, task: str) -> dict:
    fd_data = {}
    for sub in subjects:
        p = PREPROC_DIR / sub / f"{sub}_task-{task}_fd.txt"
        if p.exists():
            fd = np.loadtxt(str(p))
            fd_data[sub] = fd
    return fd_data


# ═══════════════════════════════════════════════════════════════════════════
# Plot functions
# ═══════════════════════════════════════════════════════════════════════════

def plot_group_spatial_map(imgs: list, task: str, method: str, out_dir: Path):
    """Average spatial maps across subjects and plot."""
    if not imgs:
        log.warning("  No spatial maps to average")
        return

    log.info(f"  Computing group mean spatial map ({len(imgs)} subjects) …")
    ref = imgs[0]
    resampled = [image.resample_to_img(img, ref, interpolation="continuous")
                 for img in imgs]
    mean_img = image.mean_img(resampled)

    out_nii = FIG_DIR / f"group_task-{task}_{method}_spatial_mean.nii.gz"
    nib.save(mean_img, str(out_nii))

    display = plotting.plot_stat_map(
        mean_img, display_mode="z", cut_coords=10,
        title=f"Group QPP spatial map | task-{task} | {method.upper()} (N={len(imgs)})",
        colorbar=True, cmap="cold_hot", threshold=0.1,
    )
    out_png = FIG_DIR / f"group_task-{task}_{method}_spatial_mean.png"
    display.savefig(str(out_png), dpi=150)
    display.close()
    log.info(f"  Group map saved → {out_png.name}")


def plot_qpp_strength(records: list, task: str, method: str, out_dir: Path):
    """Bar chart of QPP mean correlation per subject."""
    subjects, strengths = [], []
    for r in records:
        if method in r and "mean_corr" in r[method]:
            subjects.append(r["subject"])
            strengths.append(r[method]["mean_corr"])

    if not subjects:
        log.warning("  No QPP strength data found")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(subjects) * 0.35), 5))
    colors = ["steelblue" if s >= 0.3 else "salmon" for s in strengths]
    bars = ax.bar(subjects, strengths, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(np.mean(strengths), color="black", linestyle="--", linewidth=1,
               label=f"Group mean = {np.mean(strengths):.3f}")
    ax.set_xlabel("Subject")
    ax.set_ylabel("QPP Mean Correlation")
    ax.set_title(f"QPP Strength per Subject | task-{task} | {method.upper()}")
    ax.set_xticklabels(subjects, rotation=90, fontsize=7)
    ax.legend(fontsize=9)
    plt.tight_layout()

    out_path = out_dir / f"group_task-{task}_{method}_qpp_strength.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    log.info(f"  QPP strength chart saved → {out_path.name}")


def plot_periodicity_summary(records: list, task: str, method: str, out_dir: Path):
    """Scatter plot of peak QPP period and frequency per subject."""
    subjects, periods, freqs = [], [], []
    for r in records:
        if method in r and "peak_period_s" in r[method]:
            subjects.append(r["subject"])
            periods.append(r[method]["peak_period_s"])
            freqs.append(r[method]["peak_freq_hz"])

    if not subjects:
        log.warning("  No periodicity data found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(range(len(subjects)), periods, color="steelblue", s=40, zorder=3)
    axes[0].axhline(np.mean(periods), color="red", linestyle="--",
                    label=f"Mean={np.mean(periods):.1f}s")
    axes[0].axhspan(10, 30, alpha=0.1, color="green", label="Typical QPP range")
    axes[0].set_xticks(range(len(subjects)))
    axes[0].set_xticklabels(subjects, rotation=90, fontsize=7)
    axes[0].set_ylabel("QPP Period (s)")
    axes[0].set_title(f"QPP Period | task-{task} | {method.upper()}")
    axes[0].legend(fontsize=8)

    axes[1].scatter(range(len(subjects)), [f * 1000 for f in freqs],
                    color="darkorange", s=40, zorder=3)
    axes[1].axhline(np.mean(freqs) * 1000, color="red", linestyle="--",
                    label=f"Mean={np.mean(freqs)*1000:.1f}mHz")
    axes[1].axhspan(50, 100, alpha=0.1, color="green", label="QPP band 50–100 mHz")
    axes[1].set_xticks(range(len(subjects)))
    axes[1].set_xticklabels(subjects, rotation=90, fontsize=7)
    axes[1].set_ylabel("QPP Frequency (mHz)")
    axes[1].set_title(f"QPP Frequency | task-{task} | {method.upper()}")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    out_path = out_dir / f"group_task-{task}_{method}_periodicity.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    log.info(f"  Periodicity summary saved → {out_path.name}")


def plot_motion_qc_summary(subjects: list, task: str, out_dir: Path):
    """Group motion QC: mean FD per subject + exclusion threshold line."""
    fd_data = load_motion_fd(subjects, task)
    if not fd_data:
        log.warning("  No FD data found for motion QC")
        return

    subs = sorted(fd_data.keys())
    mean_fds = [fd_data[s].mean() for s in subs]
    max_fds = [fd_data[s].max() for s in subs]

    fig, ax = plt.subplots(figsize=(max(8, len(subs) * 0.35), 5))
    x = np.arange(len(subs))
    ax.bar(x, mean_fds, color="steelblue", alpha=0.8, label="Mean FD")
    ax.errorbar(x, mean_fds, yerr=[np.zeros(len(subs)), np.array(max_fds) - np.array(mean_fds)],
                fmt="none", color="navy", capsize=3, linewidth=0.8, label="Max FD")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Exclusion (0.5mm)")
    ax.axhline(0.2, color="orange", linestyle="--", linewidth=0.8, label="High-quality (0.2mm)")
    ax.set_xticks(x)
    ax.set_xticklabels(subs, rotation=90, fontsize=7)
    ax.set_ylabel("Framewise Displacement (mm)")
    ax.set_title(f"Motion QC Summary | task-{task}")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    out_path = out_dir / f"group_task-{task}_motion_qc.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    log.info(f"  Motion QC summary saved → {out_path.name}")

    # Flag high-motion subjects
    flagged = [s for s, fd in zip(subs, mean_fds) if fd > 0.5]
    if flagged:
        log.warning(f"  High-motion subjects (mean FD>0.5mm): {flagged}")
    return flagged


def plot_rest_vs_fe_comparison(subjects: list, method: str, out_dir: Path):
    """Scatter plot: QPP strength rest vs. emotion task per subject."""
    rest_strengths, fe_strengths, paired_subs = [], [], []

    for sub in subjects:
        rest_p = QPP_DIR / sub / f"{sub}_task-rest_qpp_summary.json"
        fe_p = QPP_DIR / sub / f"{sub}_task-fe_qpp_summary.json"
        if rest_p.exists() and fe_p.exists():
            with open(rest_p) as f:
                r_rest = json.load(f)
            with open(fe_p) as f:
                r_fe = json.load(f)
            if method in r_rest and method in r_fe:
                if "mean_corr" in r_rest[method] and "mean_corr" in r_fe[method]:
                    rest_strengths.append(r_rest[method]["mean_corr"])
                    fe_strengths.append(r_fe[method]["mean_corr"])
                    paired_subs.append(sub)

    if len(paired_subs) < 2:
        log.warning("  Not enough paired data for rest vs. emotion comparison")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(rest_strengths, fe_strengths, color="steelblue", s=50, zorder=3)
    lim_min = min(min(rest_strengths), min(fe_strengths)) * 0.9
    lim_max = max(max(rest_strengths), max(fe_strengths)) * 1.1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=0.8,
            label="Identity (rest = emotion)")
    for i, sub in enumerate(paired_subs):
        ax.annotate(sub.replace("sub-", ""), (rest_strengths[i], fe_strengths[i]),
                    fontsize=6, ha="center", va="bottom")
    ax.set_xlabel("QPP Strength (rest)")
    ax.set_ylabel("QPP Strength (emotion task)")
    ax.set_title(f"QPP Strength: Resting-State vs. Emotion Task | {method.upper()}")
    ax.legend(fontsize=8)
    plt.tight_layout()

    out_path = out_dir / f"group_{method}_rest_vs_fe.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    log.info(f"  Rest vs. emotion comparison saved → {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Group visualization for QPP results")
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--tasks", nargs="+", default=["rest", "fe"])
    parser.add_argument("--method", default="swc", choices=["swc", "seed"],
                        help="QPP method to visualize")
    args = parser.parse_args()

    if args.subjects:
        subjects = args.subjects
    else:
        subjects = sorted([d.name for d in QPP_DIR.iterdir()
                           if d.is_dir() and d.name.startswith("sub-")])

    if not subjects:
        log.error("No processed subjects found in qpp_analysis/. Run 02_qpp_detection.py first.")
        sys.exit(1)

    log.info(f"Visualizing {len(subjects)} subject(s)  method={args.method}")

    for task in args.tasks:
        log.info(f"\n── Task: {task} ──")
        records = load_summaries(subjects, task)
        spatial_imgs = load_spatial_maps(subjects, task, args.method)

        plot_group_spatial_map(spatial_imgs, task, args.method, FIG_DIR)
        plot_qpp_strength(records, task, args.method, FIG_DIR)
        plot_periodicity_summary(records, task, args.method, FIG_DIR)
        plot_motion_qc_summary(subjects, task, FIG_DIR)

    # Cross-task comparison (only if both tasks were processed)
    if "rest" in args.tasks and "fe" in args.tasks:
        plot_rest_vs_fe_comparison(subjects, args.method, FIG_DIR)

    log.info("\nVisualization complete.  Figures → outputs/figures/")


if __name__ == "__main__":
    main()
