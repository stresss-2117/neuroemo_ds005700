"""
Script 9 (v2): Multimodal Overlap / Cohesion Maps — Emotion-Cognition Interplay
==================================================================================
CORRECTED DESIGN vs the first version of this script: your ReHo, fALFF, and
GLM group-level significance maps are ALREADY COMPUTED and saved to disk by
04_reho_analysis.py / 05_activation_contrast.py / 06_glm_analysis.py (this
is visible from your existing 07_cognition_emotion.py, which reads them
directly). Re-extracting per-subject values and re-running paired t-tests
for those three modalities — like the first version of this script did --
was redundant and risked a second, inconsistent significance decision on
top of an already-validated one.

This version:
  - ReHo / fALFF / GLM  -> load the EXISTING group-level FDR-thresholded
    maps directly (voxelwise, already corrected). No new stats computed.
  - QPP                 -> the one modality with no precomputed group
    contrast. Computed here as a voxelwise paired t-test (fe vs rest),
    vectorized, BH-FDR corrected -- same logic as your old
    compute_qpp_contrast(), but with scipy removed (manual incomplete-beta,
    vectorized over voxels) so it survives the Windows scipy.stats DLL
    issue without needing scipy at all.
  - All 4 maps resampled to a common reference, then summarized at
    Schaefer-400 ROI level (fraction of in-ROI voxels significant),
    then rolled up to Yeo7 network level via each Schaefer parcel's
    encoded network name.

Usage:
  python scripts/09_multimodal_overlap_v2.py
"""

import os
import math
import json
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from nilearn import datasets, image, plotting
from nilearn.image import resample_to_img

BASE_DIR = Path(".")
REHO_DIR = BASE_DIR / "reho_analysis"
ACT_DIR = BASE_DIR / "activation_contrast"
GLM_DIR = BASE_DIR / "glm_analysis"
QPP_RESULTS_DIR = BASE_DIR / "results"
OUT_DIR = BASE_DIR / "results" / "figures" / "multimodal_overlap_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FDR_ALPHA = 0.05
ALL_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 41)]
EMOTIONS = ["calm", "afraid", "delighted", "depressed", "excited"]

# An ROI counts as "significant" for a modality if at least this fraction
# of its in-ROI voxels are nonzero in that modality's thresholded map.
# Heuristic, not a validated cutoff -- inspect roi_coverage.csv and adjust.
ROI_COVERAGE_THRESHOLD = 0.10
NEAR_UNANIMOUS_THRESHOLD = 3   # overlap_count >= this -> "cognitive" candidate


# ══════════════════════════════════════════════════════════════
# Vectorized, scipy-free paired t-test (for the QPP voxelwise contrast)
# ══════════════════════════════════════════════════════════════

def _betainc_vec(a, b, x, iters=100):
    """Regularized incomplete beta function, vectorized over a numpy array x."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    mask_full = x >= 1
    mask_zero = x <= 0
    mask_mid = ~mask_full & ~mask_zero
    out[mask_full] = 1.0
    if not mask_mid.any():
        return out

    xin = x[mask_mid]
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
              + a * np.log(xin) + b * np.log(1 - xin))
    front = np.exp(lbeta) / a
    f = np.ones_like(xin)
    c = np.ones_like(xin)
    d = np.zeros_like(xin)

    for i in range(iters):
        m = i // 2
        if i == 0:
            numerator = np.ones_like(xin)
        elif i % 2 == 0:
            numerator = (m * (b - m) * xin) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            numerator = -((a + m) * (a + b + m) * xin) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        d = np.where(np.abs(d) < 1e-30, 1e-30, d)
        d = 1.0 / d
        c = 1.0 + numerator / c
        c = np.where(np.abs(c) < 1e-30, 1e-30, c)
        f = f * d * c

    result = front * (f - 1)
    thresh = (a + 1) / (a + b + 2)
    result = np.where(xin < thresh, result, 1 - result)
    out[mask_mid] = result
    return out


def ttest_rel_vec(a_stack, b_stack):
    """Paired t-test, vectorized over voxels. a_stack/b_stack: (n_subjects, n_voxels)."""
    diff = a_stack - b_stack
    n = diff.shape[0]
    mean_d = diff.mean(axis=0)
    std_d = diff.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(std_d > 1e-12, mean_d / (std_d / np.sqrt(n)), 0.0)
    df = n - 1
    x = df / (df + t ** 2)
    p = _betainc_vec(df / 2, 0.5, x)
    return t, p


def bh_fdr_voxelwise(p_map, mask, alpha=FDR_ALPHA):
    p_vals = p_map[mask]
    n = len(p_vals)
    sorted_idx = np.argsort(p_vals)
    sorted_p = p_vals[sorted_idx]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    below = sorted_p <= thresholds
    if not below.any():
        return np.zeros_like(mask, dtype=bool)
    cutoff = sorted_p[np.where(below)[0].max()]
    sig = np.zeros_like(mask, dtype=bool)
    sig[mask] = p_vals <= cutoff
    return sig


def compute_qpp_contrast(subjects, ref_img):
    """Paired t-test: per-subject QPP (fe vs rest), FDR-corrected. No scipy."""
    rest_imgs, fe_imgs, used = [], [], []
    for sub in subjects:
        r = QPP_RESULTS_DIR / "rest" / f"{sub}_qpp_template_MNI_ANTs.nii.gz"
        f = QPP_RESULTS_DIR / "fe" / f"{sub}_qpp_template_MNI_ANTs.nii.gz"
        if r.exists() and f.exists():
            rest_imgs.append(nib.load(str(r)))
            fe_imgs.append(nib.load(str(f)))
            used.append(sub)

    if len(used) < 2:
        print(f"  Not enough paired QPP subjects found ({len(used)}). Skipping QPP contrast.")
        return None

    print(f"  QPP paired t-test: N={len(used)}")

    def mid_frame(img):
        d = img.get_fdata()
        return d[:, :, :, d.shape[3] // 2]

    def resample(imgs):
        return np.stack(
            [image.resample_to_img(
                nib.Nifti1Image(mid_frame(img), img.affine), ref_img,
                interpolation="continuous").get_fdata(dtype=np.float32)
             for img in imgs], axis=0)

    rest_stack = resample(rest_imgs)
    fe_stack = resample(fe_imgs)

    ref_data = ref_img.get_fdata()
    mask = ref_data.astype(bool)

    diff_masked_rest = rest_stack[:, mask]
    diff_masked_fe = fe_stack[:, mask]
    t_vals, p_vals = ttest_rel_vec(diff_masked_fe, diff_masked_rest)

    t_map = np.zeros(ref_img.shape[:3], dtype=np.float32)
    p_map = np.ones(ref_img.shape[:3], dtype=np.float32)
    t_map[mask] = t_vals
    p_map[mask] = p_vals

    sig_mask = bh_fdr_voxelwise(p_map, mask)
    t_fdr = t_map.copy()
    t_fdr[~sig_mask] = 0.0

    print(f"  QPP significant voxels (FDR q<{FDR_ALPHA}): {sig_mask.sum()}")
    return nib.Nifti1Image(t_fdr, ref_img.affine)


# ══════════════════════════════════════════════════════════════
# Load existing precomputed group maps (ReHo / fALFF / GLM) — NO new stats
# ══════════════════════════════════════════════════════════════

def load_binary_mask(path, ref_img, label):
    if not Path(path).exists():
        print(f"  [{label}] MISSING: {path}")
        return None
    img = nib.load(str(path))
    img_res = resample_to_img(img, ref_img, interpolation="nearest")
    data = img_res.get_fdata()
    binary = (data != 0).astype(np.uint8)
    print(f"  [{label}] {binary.sum()} significant voxels (from precomputed map)")
    return binary


def combine_glm_emotions(ref_img):
    """OR across all 5 emotion contrasts = 'any emotion activation'."""
    combined = None
    for emo in EMOTIONS:
        path = GLM_DIR / f"group_{emo}_vs_noise_z_fdr.nii.gz"
        b = load_binary_mask(path, ref_img, f"GLM-{emo}")
        if b is None:
            continue
        combined = b if combined is None else np.logical_or(combined, b)
    if combined is None:
        return None
    print(f"  [GLM combined] {combined.sum()} voxels significant in >=1 emotion")
    return combined.astype(np.uint8)


# ══════════════════════════════════════════════════════════════
# Schaefer-400 atlas + Yeo7 network parsing
# ══════════════════════════════════════════════════════════════

YEO7_NAME_MAP = {
    "Vis": "Visual", "SomMot": "Somatomotor", "DorsAttn": "DorsalAttention",
    "SalVentAttn": "VentralAttention", "Limbic": "Limbic",
    "Cont": "Frontoparietal", "Default": "Default",
}


def get_schaefer_400():
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=2)
    atlas_img = nib.load(schaefer.maps)
    labels = [l.decode() if isinstance(l, bytes) else l for l in schaefer.labels]
    label_map = {i + 1: name for i, name in enumerate(labels)}
    return atlas_img, label_map


def parcel_to_yeo7(parcel_name):
    for key, network in YEO7_NAME_MAP.items():
        if f"_{key}_" in parcel_name or parcel_name.endswith(f"_{key}"):
            return network
    return "Unknown"


def rois_significant_from_mask(binary_mask, ref_img, atlas_img, label_map,
                                coverage_threshold=ROI_COVERAGE_THRESHOLD):
    """Resample a voxelwise binary significance mask onto the Schaefer atlas,
    then flag each ROI significant if >= coverage_threshold of its voxels
    are nonzero in the mask."""
    mask_img = nib.Nifti1Image(binary_mask.astype(np.float32), ref_img.affine)
    atlas_res = resample_to_img(atlas_img, mask_img, interpolation="nearest")
    atlas_data = atlas_res.get_fdata()
    if atlas_data.ndim == 4:
        atlas_data = atlas_data[..., 0]
    atlas_data = atlas_data.astype(int)

    sig_set = set()
    coverage_rows = []
    for roi_id, roi_name in label_map.items():
        roi_mask = (atlas_data == roi_id)
        n_vox = roi_mask.sum()
        if n_vox == 0:
            continue
        frac_sig = binary_mask[roi_mask].mean()
        coverage_rows.append({"ROI": roi_name, "n_voxels": int(n_vox), "frac_significant": frac_sig})
        if frac_sig >= coverage_threshold:
            sig_set.add(roi_name)
    return sig_set, pd.DataFrame(coverage_rows)


# ══════════════════════════════════════════════════════════════
# Overlap + Yeo7 network summary
# ══════════════════════════════════════════════════════════════

def compute_overlap(label_map, sig_sets):
    rows = []
    for roi_id, roi_name in label_map.items():
        overlap_count = sum(1 for s in sig_sets.values() if roi_name in s)
        rows.append({
            "ROI": roi_name, "roi_id": roi_id, "overlap_count": overlap_count,
            "Yeo7Network": parcel_to_yeo7(roi_name),
            **{f"sig_{mod}": (roi_name in s) for mod, s in sig_sets.items()},
        })
    return pd.DataFrame(rows)


def summarize_by_network(overlap_df, n_modalities):
    rows = []
    for network in sorted(overlap_df["Yeo7Network"].unique()):
        sub = overlap_df[overlap_df.Yeo7Network == network]
        if len(sub) == 0:
            continue
        pct_high_agreement = (sub["overlap_count"] >= NEAR_UNANIMOUS_THRESHOLD).mean() * 100
        pct_single_modality = (sub["overlap_count"] == 1).mean() * 100
        rows.append({
            "Network": network, "N_ROIs": len(sub),
            "MeanOverlap": sub["overlap_count"].mean(),
            "PctHighAgreement": pct_high_agreement,
            "PctSingleModality": pct_single_modality,
        })
    df = pd.DataFrame(rows).sort_values("MeanOverlap", ascending=False)
    df["Classification"] = df["PctHighAgreement"].apply(
        lambda p: "Cognitive (domain-general)" if p > 10 else "Emotion/modality-specific")
    return df


def build_overlap_image(overlap_df, atlas_img, label_map):
    atlas_data = atlas_img.get_fdata()
    if atlas_data.ndim == 4:
        atlas_data = atlas_data[..., 0]
    out = np.zeros_like(atlas_data)
    name_to_count = dict(zip(overlap_df["ROI"], overlap_df["overlap_count"]))
    for roi_id, name in label_map.items():
        out[atlas_data == roi_id] = name_to_count.get(name, 0)
    return nib.Nifti1Image(out, affine=atlas_img.affine)


def plot_overlap_brain(overlap_img, n_modalities, out_path):
    cmap = mpl.colormaps["YlOrRd"].resampled(n_modalities + 1)
    display = plotting.plot_stat_map(
        overlap_img, display_mode="z", cut_coords=8, black_bg=True,
        title=f"Cross-Modality Overlap (0-{n_modalities} modalities agree)",
        colorbar=True, cmap=cmap, threshold=0.5, vmax=n_modalities, annotate=True,
    )
    display.savefig(str(out_path), dpi=200)
    display.close()


def plot_network_bar(df, out_path, n_modalities):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d73027" if c == "Cognitive (domain-general)" else "#4575b4"
              for c in df["Classification"]]
    ax.barh(df["Network"], df["MeanOverlap"], color=colors, edgecolor="white")
    ax.set_xlabel(f"Mean overlap count (0-{n_modalities} modalities)")
    ax.set_title("Yeo7 Network: Cross-Modality Agreement (Schaefer-400 -> Yeo7)\n"
                  "Red = candidate cognitive/domain-general network")
    ax.axvline(NEAR_UNANIMOUS_THRESHOLD, color="black", linestyle="--", linewidth=0.8,
               label=f"Near-unanimous threshold ({NEAR_UNANIMOUS_THRESHOLD}/{n_modalities})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    reho_path = REHO_DIR / "group_rest-vs-fe_reho_t_thr.nii.gz"
    if not reho_path.exists():
        raise FileNotFoundError(f"{reho_path} not found — run 04_reho_analysis.py first.")
    ref_img = nib.load(str(reho_path))

    print("Loading precomputed group maps...")
    binary_masks = {}
    binary_masks["ReHo"] = load_binary_mask(reho_path, ref_img, "ReHo")
    binary_masks["fALFF"] = load_binary_mask(
        ACT_DIR / "group_fe_vs_rest_t_fdr.nii.gz", ref_img, "fALFF")
    binary_masks["GLM"] = combine_glm_emotions(ref_img)

    print("\nComputing QPP contrast (no precomputed map exists for this one)...")
    qpp_contrast_img = compute_qpp_contrast(ALL_SUBJECTS, ref_img)
    if qpp_contrast_img is not None:
        nib.save(qpp_contrast_img, str(OUT_DIR / "qpp_rest_vs_fe_contrast.nii.gz"))
        binary_masks["QPP"] = (qpp_contrast_img.get_fdata() != 0).astype(np.uint8)
    else:
        binary_masks["QPP"] = None

    active_masks = {k: v for k, v in binary_masks.items() if v is not None}
    n_modalities = len(active_masks)
    print(f"\nUsing {n_modalities} modalities: {list(active_masks.keys())}")
    if n_modalities < 2:
        raise RuntimeError("Need at least 2 modalities to compute a meaningful overlap.")

    print("\nLoading Schaefer-400 atlas and mapping voxelwise masks to ROIs...")
    atlas_img, label_map = get_schaefer_400()

    sig_sets = {}
    for modality, mask in active_masks.items():
        sig_set, coverage_df = rois_significant_from_mask(mask, ref_img, atlas_img, label_map)
        sig_sets[modality] = sig_set
        coverage_df.to_csv(OUT_DIR / f"{modality.lower()}_roi_coverage.csv", index=False)
        print(f"  [{modality}] {len(sig_set)} / {len(label_map)} ROIs >= "
              f"{ROI_COVERAGE_THRESHOLD*100:.0f}% voxel coverage")

    overlap_df = compute_overlap(label_map, sig_sets)
    overlap_df.to_csv(OUT_DIR / "schaefer400_overlap_table.csv", index=False)

    network_summary = summarize_by_network(overlap_df, n_modalities)
    network_summary.to_csv(OUT_DIR / "yeo7_network_overlap_summary.csv", index=False)
    print("\nYeo7 network overlap summary:")
    print(network_summary.to_string(index=False))

    overlap_img = build_overlap_image(overlap_df, atlas_img, label_map)
    nib.save(overlap_img, str(OUT_DIR / "overlap_count.nii.gz"))
    plot_overlap_brain(overlap_img, n_modalities, OUT_DIR / "overlap_count_axial.png")
    plot_network_bar(network_summary, OUT_DIR / "yeo7_network_overlap_bar.png", n_modalities)

    cognitive_networks = network_summary[
        network_summary["Classification"] == "Cognitive (domain-general)"]["Network"].tolist()
    emotion_specific_networks = network_summary[
        network_summary["Classification"] != "Cognitive (domain-general)"]["Network"].tolist()

    summary = {
        "n_modalities_used": n_modalities, "modalities": list(active_masks.keys()),
        "roi_coverage_threshold": ROI_COVERAGE_THRESHOLD,
        "candidate_cognitive_networks": cognitive_networks,
        "candidate_emotion_specific_networks": emotion_specific_networks,
        "note": ("'Cognitive/domain-general' = Yeo7 networks where >10% of their "
                 "Schaefer-400 ROIs show near-unanimous (>=3/4) modality agreement. "
                 "ReHo/fALFF/GLM significance comes from your existing precomputed "
                 "group-level FDR maps; QPP significance is computed fresh here. "
                 "This is a descriptive heuristic, not a validated cutoff."),
    }
    with open(OUT_DIR / "overlap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nCandidate cognitive/domain-general networks: {cognitive_networks}")
    print(f"Candidate emotion/modality-specific networks: {emotion_specific_networks}")
    print(f"\nDone. All outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()