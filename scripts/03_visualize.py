"""
Group-level visualization for NeuroEmo QPP analysis.

Generates:
  1. Group QPP spatial map (mean across subjects)
  2. QPP strength (mean correlation) bar chart per subject
  3. QPP periodicity summary (peak frequency + period)
  4. Motion QC summary across all subjects
  5. Group-level spatial map overlaid on MNI template

Usage:
  python scripts/03_visualize.py [--tasks rest emotion] [--method swc seed] [--subjects sub-01 sub-02]
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

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# Updated directory mapping to match actual pipeline structure
PREPROC_DIR = BASE_DIR / "data" / "processed"
QPP_DIR = BASE_DIR / "results"
FIG_DIR = BASE_DIR / "outputs" / "figures"
LOG_DIR = BASE_DIR / "outputs" / "logs"

FIG_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

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
        # Check both subject subfolders and task subfolders in results/
        paths_to_check = [
            QPP_DIR / sub / f"{sub}_task-{task}_qpp_summary.json",
            QPP_DIR / task / f"{sub}_task-{task}_qpp_summary.json",
            QPP_DIR / f"{sub}_task-{task}_qpp_summary.json"
        ]
        found = False
        for p in paths_to_check:
            if p.exists():
                with open(p) as f:
                    records.append(json.load(f))
                found = True
                break
        if not found:
            log.warning(f"  Missing summary for {sub} task-{task}")
    return records


def load_spatial_maps(subjects: list, task: str, method: str) -> list:
    imgs = []
    for sub in subjects:
        filename = f"{sub}_task-{task}_swc_qpp_template.nii.gz" if method == "swc" else f"{sub}_task-{task}_seed_corr_map.nii.gz"
        
        paths_to_check = [
            QPP_DIR / sub / filename,
            QPP_DIR / task / filename,
            QPP_DIR / filename
        ]
        found = False
        for p in paths_to_check:
            if p.exists():
                imgs.append(nib.load(str(p)))
                found = True
                break
        if not found:
            log.warning(f"  Missing map for {sub} task-{task} ({method})")
    return imgs


def load_motion_fd(subjects: list, task: str) -> dict:
    fd_data = {}
    for sub in subjects:
        paths_to_check = [
            PREPROC_DIR / sub / f"{sub}_task-{task}_fd.txt",
            PREPROC_DIR / f"{sub}_task-{task}_fd.txt"
        ]
        for p in paths_to_check:
            if p.exists():
                fd = np.loadtxt(str(p))
                fd_data[sub] = fd
                break
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
    resampled = [image.resample_to_img(img, ref, interpolation="continuous") for img in imgs]
    mean_img = image.mean_img(resampled)

    out_nii = out_dir / f"group_task-{task}_{method}_spatial_mean.nii.gz"
    nib.save(mean_img, str(out_nii))

    display = plotting.plot_stat_map(
        mean_img, display_mode="z", cut_coords=10,
        title=f"Group QPP spatial map | task-{task} | {method.upper()} (N={len(imgs)})",
        colorbar=True, cmap="cold_hot", threshold=0.1,
    )
    out_png = out_dir / f"group_task-{task}_{method}_spatial_mean.png"
    display.savefig(str(out_png), dpi=150)
    display.close()
    log.info(f"  Group map saved → {out_png.name}")


def plot_qpp_strength(records: list, task: str, method: str, out_dir: Path):
    """Bar chart of QPP mean correlation per subject."""
    subjects, strengths = [], []
    for r in records:
        if method in r and "mean_corr" in r[method]:
            subjects.append(r.get("subject", "unknown"))
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
            subjects.append(r.get("subject", "unknown"))
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


def plot_rest_vs_fe_comparison(subjects: list, method: str, out_dir: Path, tasks: list):
    """Scatter plot: QPP strength rest vs. emotion task per subject."""
    t1, t2 = tasks[0], tasks[1]
    strengths_t1, strengths_t2, paired_subs = [], [], []

    for sub in subjects:
        p1 = QPP_DIR / f"{sub}_task-{t1}_qpp_summary.json"
        p2 = QPP_DIR / f"{sub}_task-{t2}_qpp_summary.json"
        
        if p1.exists() and p2.exists():
            with open(p1) as f:
                r1 = json.load(f)
            with open(p2) as f:
                r2 = json.load(f)
            if method in r1 and method in r2:
                if "mean_corr" in r1[method] and "mean_corr" in r2[method]:
                    strengths_t1.append(r1[method]["mean_corr"])
                    strengths_t2.append(r2[method]["mean_corr"])
                    paired_subs.append(sub)

    if len(paired_subs) < 2:
        log.warning("  Not enough paired data for rest vs. emotion comparison")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(strengths_t1, strengths_t2, color="steelblue", s=50, zorder=3)
    lim_min = min(min(strengths_t1), min(strengths_t2)) * 0.9
    lim_max = max(max(strengths_t1), max(strengths_t2)) * 1.1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=0.8,
            label="Identity")
    for i, sub in enumerate(paired_subs):
        ax.annotate(sub.replace("sub-", ""), (strengths_t1[i], strengths_t2[i]),
                    fontsize=6, ha="center", va="bottom")
    ax.set_xlabel(f"QPP Strength ({t1})")
    ax.set_ylabel(f"QPP Strength ({t2})")
    ax.set_title(f"QPP Strength: {t1.upper()} vs. {t2.upper()} | {method.upper()}")
    ax.legend(fontsize=8)
    plt.tight_layout()

    out_path = out_dir / f"group_{method}_rest_vs_fe.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    log.info(f"  Task comparison saved → {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Group visualization for QPP results")
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--tasks", nargs="+", default=["rest", "emotion"])
    parser.add_argument("--method", default="swc", choices=["swc", "seed"],
                        help="QPP method to visualize")
    args = parser.parse_args()

    if args.subjects:
        subjects = args.subjects
    else:
        # Check both results/ and data/processed/ for subjects starting with sub-
        sub_dirs = set()
        if QPP_DIR.exists():
            sub_dirs.update([d.name for d in QPP_DIR.iterdir() if d.is_dir() and d.name.startswith("sub-")])
            sub_dirs.update([f.name.split("_")[0] for f in QPP_DIR.glob("sub-*")])
        if PREPROC_DIR.exists():
            sub_dirs.update([d.name for d in PREPROC_DIR.iterdir() if d.is_dir() and d.name.startswith("sub-")])
            sub_dirs.update([f.name.split("_")[0] for f in PREPROC_DIR.glob("sub-*")])
            
        subjects = sorted(list(sub_dirs))

    if not subjects:
        log.error("No processed subjects found in results/ or data/processed/. Check your data directories.")
        sys.exit(1)

    log.info(f"Visualizing {len(subjects)} subject(s): {subjects} | method={args.method}")

    for task in args.tasks:
        log.info(f"\n── Task: {task} ──")
        records = load_summaries(subjects, task)
        spatial_imgs = load_spatial_maps(subjects, task, args.method)

        plot_group_spatial_map(spatial_imgs, task, args.method, FIG_DIR)
        plot_qpp_strength(records, task, args.method, FIG_DIR)
        plot_periodicity_summary(records, task, args.method, FIG_DIR)
        plot_motion_qc_summary(subjects, task, FIG_DIR)

    # Cross-task comparison (if multiple tasks are supplied)
    if len(args.tasks) >= 2:
        plot_rest_vs_fe_comparison(subjects, args.method, FIG_DIR, args.tasks)

    log.info("\nVisualization complete. Figures saved → outputs/figures/")


if __name__ == "__main__":
    main()