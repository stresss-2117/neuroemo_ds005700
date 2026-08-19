"""
QPP-Only Cognition-Emotion Interplay Analysis (Yeo7)
=======================================================
Uses ONLY your own QPP pipeline (extract_qpp_03.py -> results/{task}/{sub}_qpp_template_MNI_ANTs.nii.gz)
-- no dependency on the TA's ReHo/fALFF/GLM/SWC-QPP outputs.

Improvement over the earlier version: instead of collapsing each subject's QPP
template to a single "middle frame" snapshot, this uses the FULL QPP cycle
(mean absolute amplitude across all window frames). This is both more
statistically sensitive AND more faithful to what a QPP actually is -- a
repeating spatiotemporal pattern across the whole window, not one instant.

Per Yeo7 network, per subject, per task:
  engagement = mean( |QPP value| ) across ALL frames in the window, within
               that network's voxels
  (using absolute value because a QPP cycle by definition swings from
   high to low -- averaging signed values across the full cycle would
   cancel out real engagement toward ~0, which is exactly what happened
   with the single-frame version)

Then, per network:
  - One-sample t-test vs 0, separately for rest and fe
  - Paired t-test, fe vs rest
  - BH-FDR correction across the 7 networks (each test family separately)
  - Classification: Cognitive (domain-general) / Emotion-specific /
    Task-specific-only / Rest-specific-only / Not reliably engaged

Also flags networks with near-zero variance across subjects (the Limbic/
orbital coverage issue identified earlier) rather than silently reporting
them as an ordinary null result.

Usage:
  python scripts/07_qpp_cognition_overlap_v2.py
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats
from nilearn import datasets, plotting
from nilearn.image import resample_to_img
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

RESULTS_DIR = "results"
OUT_DIR = "results/figures/yeo7_overlap"
os.makedirs(OUT_DIR, exist_ok=True)

ALL_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 41)]
TASKS = ["rest", "fe"]
FDR_ALPHA = 0.05

# A network is flagged as "low coverage" (unreliable) if more than this
# fraction of subjects have near-zero values (suggests atlas/resampling
# coverage problems rather than genuine null signal -- e.g. Yeo7 Limbic)
LOW_COVERAGE_FRACTION_THRESHOLD = 0.5
NEAR_ZERO_ABS_THRESHOLD = 1e-4


# ══════════════════════════════════════════════════════════════
# Yeo7 atlas
# ══════════════════════════════════════════════════════════════

def get_yeo7():
    yeo = datasets.fetch_atlas_yeo_2011(n_networks=7, thickness="thick")
    atlas_img = nib.load(yeo.maps)
    names = ["Visual", "Somatomotor", "DorsalAttention", "VentralAttention",
              "Limbic", "Frontoparietal", "Default"]
    label_map = {i + 1: name for i, name in enumerate(names)}
    return atlas_img, label_map


# ══════════════════════════════════════════════════════════════
# Per-subject, full-cycle Yeo7 QPP extraction
# ══════════════════════════════════════════════════════════════

def extract_subject_network_value_full_cycle(qpp_path, atlas_data, roi_id):
    """
    Mean absolute QPP amplitude across ALL frames of the window, within
    one Yeo7 network's voxels. Using |value| because a QPP cycle swings
    from high to low by definition -- signed averaging across the full
    cycle would wash out real engagement toward ~0.
    """
    img = nib.load(qpp_path)
    data = img.get_fdata()
    roi_mask = (atlas_data == roi_id)
    if roi_mask.sum() == 0:
        return np.nan

    if data.ndim == 4:
        # mean |amplitude| per frame within the ROI, then averaged across frames
        per_frame = [np.abs(data[..., f][roi_mask]).mean() for f in range(data.shape[-1])]
        return float(np.mean(per_frame))
    else:
        return float(np.abs(data[roi_mask]).mean())


def build_subject_network_table(atlas_img, label_map):
    rows = []
    for task in TASKS:
        for sub in ALL_SUBJECTS:
            path = f"{RESULTS_DIR}/{task}/{sub}_qpp_template_MNI_ANTs.nii.gz"
            if not os.path.exists(path):
                continue

            ref_img = nib.load(path)
            ref_data = ref_img.get_fdata()
            ref_3d = nib.Nifti1Image(
                ref_data[..., 0] if ref_data.ndim == 4 else ref_data, ref_img.affine)

            atlas_res = resample_to_img(atlas_img, ref_3d, interpolation="nearest")
            atlas_data = atlas_res.get_fdata()
            if atlas_data.ndim == 4:
                atlas_data = atlas_data[..., 0]
            atlas_data = atlas_data.astype(int)

            for roi_id, name in label_map.items():
                val = extract_subject_network_value_full_cycle(path, atlas_data, roi_id)
                rows.append({"Subject": sub, "Task": task, "Network": name, "Value": val})

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/subject_network_qpp_values_fullcycle.csv", index=False)
    print(f"Saved per-subject network table -> {OUT_DIR}/subject_network_qpp_values_fullcycle.csv")
    return df


# ══════════════════════════════════════════════════════════════
# Coverage check (flag Limbic-style artifacts)
# ══════════════════════════════════════════════════════════════

def flag_low_coverage_networks(df, label_map):
    flagged = []
    for name in label_map.values():
        vals = df[df.Network == name]["Value"].dropna()
        if len(vals) == 0:
            flagged.append(name)
            continue
        frac_near_zero = (vals.abs() < NEAR_ZERO_ABS_THRESHOLD).mean()
        if frac_near_zero > LOW_COVERAGE_FRACTION_THRESHOLD:
            print(f"  [COVERAGE WARNING] '{name}': {frac_near_zero*100:.0f}% of values "
                  f"are near-zero (<{NEAR_ZERO_ABS_THRESHOLD}) -- likely an atlas/resampling "
                  f"coverage artifact, not a genuine null result. Treat with caution.")
            flagged.append(name)
    return flagged


# ══════════════════════════════════════════════════════════════
# Statistics per network
# ══════════════════════════════════════════════════════════════

def bh_fdr(pvals, alpha=FDR_ALPHA):
    pvals = np.array(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = (np.arange(1, n + 1) / n) * alpha
    passed = ranked <= thresh
    if not passed.any():
        return np.zeros(n, dtype=bool)
    cutoff = ranked[np.where(passed)[0].max()]
    return pvals <= cutoff


def run_network_stats(df, label_map, low_coverage_networks):
    rows = []
    for name in label_map.values():
        rest_vals = df[(df.Task == "rest") & (df.Network == name)].set_index("Subject")["Value"]
        fe_vals = df[(df.Task == "fe") & (df.Network == name)].set_index("Subject")["Value"]

        common = rest_vals.index.intersection(fe_vals.index)
        rest_v = rest_vals.loc[common].dropna()
        fe_v = fe_vals.loc[common].dropna()
        common2 = rest_v.index.intersection(fe_v.index)
        rest_v = rest_v.loc[common2].values
        fe_v = fe_v.loc[common2].values

        if len(rest_v) < 2:
            continue

        t_rest, p_rest = stats.ttest_1samp(rest_v, 0)
        t_fe, p_fe = stats.ttest_1samp(fe_v, 0)
        t_diff, p_diff = stats.ttest_rel(fe_v, rest_v)

        rows.append({
            "Network": name, "N": len(rest_v),
            "Rest_mean": rest_v.mean(), "Rest_t": t_rest, "Rest_p": p_rest,
            "FE_mean": fe_v.mean(), "FE_t": t_fe, "FE_p": p_fe,
            "Diff_t": t_diff, "Diff_p": p_diff,
            "LowCoverage": name in low_coverage_networks,
        })

    res = pd.DataFrame(rows)
    res["Rest_sig"] = bh_fdr(res["Rest_p"].values)
    res["FE_sig"] = bh_fdr(res["FE_p"].values)
    res["Diff_sig"] = bh_fdr(res["Diff_p"].values)

    def classify(row):
        if row["LowCoverage"]:
            return "Low atlas coverage (unreliable)"
        elif row["Diff_sig"]:
            return "Emotion-modulated (FE != Rest, FDR-corrected)"
        elif row["Diff_p"] < 0.10:
            return "Trend toward emotion-modulation (uncorrected)"
        else:
            return "No evidence of state-dependent difference"

    res["Classification"] = res.apply(classify, axis=1)
    res = res.sort_values("Diff_t", key=abs, ascending=False)
    res.to_csv(f"{OUT_DIR}/yeo7_network_stats_fullcycle.csv", index=False)
    print("\nNetwork classification (full-cycle amplitude):")
    print(res[["Network", "Rest_mean", "FE_mean", "Diff_t", "Diff_p", "Classification"]]
          .to_string(index=False))
    return res


# ══════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════

CLASS_COLORS = {
    "Emotion-modulated (FE != Rest, FDR-corrected)": "#d73027",
    "Trend toward emotion-modulation (uncorrected)": "#fc8d59",
    "No evidence of state-dependent difference": "#bababa",
    "Low atlas coverage (unreliable)": "#404040",
}

def plot_classification_bar(res, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [CLASS_COLORS[c] for c in res["Classification"]]
    ax.barh(res["Network"], res["Diff_t"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Paired t-statistic (FE - Rest QPP engagement, full-cycle |amplitude|)")
    ax.set_title("Yeo7 Network QPP Engagement: Rest vs. Emotion Task\n"
                 "(color = classification; full QPP cycle used, not single frame)")
    handles = [Patch(color=c, label=k) for k, c in CLASS_COLORS.items()]
    ax.legend(handles=handles, fontsize=7, loc="best")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_classification_brain(res, atlas_img, label_map, out_path):
    class_to_int = {c: i + 1 for i, c in enumerate(CLASS_COLORS.keys())}
    name_to_class = dict(zip(res["Network"], res["Classification"]))

    atlas_data = atlas_img.get_fdata()
    if atlas_data.ndim == 4:
        atlas_data = atlas_data[..., 0]

    out_data = np.zeros_like(atlas_data)
    for roi_id, name in label_map.items():
        cls = name_to_class.get(name)
        if cls is not None:
            out_data[atlas_data == roi_id] = class_to_int[cls]

    out_img = nib.Nifti1Image(out_data, affine=atlas_img.affine)
    cmap = ListedColormap(["#000000"] + list(CLASS_COLORS.values()))

    display = plotting.plot_roi(
        out_img, display_mode="z", cut_coords=8,
        title="Yeo7 Networks: QPP Cognitive vs. Emotion-Specific (full-cycle)",
        cmap=cmap, colorbar=False, black_bg=True, annotate=True,
    )
    display.savefig(out_path, dpi=200)
    display.close()
    print(f"Saved -> {out_path}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    atlas_img, label_map = get_yeo7()

    print("Extracting per-subject Yeo7 QPP values (FULL CYCLE, rest + fe)...")
    df = build_subject_network_table(atlas_img, label_map)

    print("\nChecking for atlas coverage artifacts (e.g. Limbic/orbital issue)...")
    low_coverage = flag_low_coverage_networks(df, label_map)

    print("\nRunning network-level statistics...")
    res = run_network_stats(df, label_map, low_coverage)

    plot_classification_bar(res, f"{OUT_DIR}/yeo7_classification_bar_fullcycle.png")
    plot_classification_brain(res, atlas_img, label_map,
                               f"{OUT_DIR}/yeo7_classification_brain_fullcycle.png")

    print(f"\nDone. All outputs in {OUT_DIR}/")