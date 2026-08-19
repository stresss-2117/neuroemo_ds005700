"""
Emotion–Cognition Interplay: Multi-Modality Overlap Analysis
===============================================================
Combines group-level rest-vs-emotion contrasts from ReHo, fALFF, QPP,
and GLM into a single "overlap count" map, viewed through the Yeo7
functional network lens.

Logic:
  - Each modality's thresholded contrast map is binarized (significant / not)
  - Binary masks are summed → overlap_count (0 to 4)
  - Voxels/networks with high overlap across ALL modalities = domain-general
    ("cognitive") — active regardless of which measure or state you use
  - Voxels/networks significant in only ONE modality = modality- or
    emotion-specific

Usage:
  python scripts/07_cognition_overlap.py
"""

import json
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from scipy import stats
from nilearn import datasets, image, plotting
import matplotlib.pyplot as plt
import matplotlib as mpl

BASE_DIR = Path(__file__).resolve().parent.parent
REHO_DIR = BASE_DIR / "reho_analysis"
ACT_DIR  = BASE_DIR / "activation_contrast"
GLM_DIR  = BASE_DIR / "glm_analysis"
QPP_RESULTS_DIR = BASE_DIR / "results"
OUT_DIR  = BASE_DIR / "results" / "figures" / "yeo7_overlap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FDR_ALPHA = 0.05
ALL_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 41)]
EMOTIONS = ["calm", "afraid", "delighted", "depressed", "excited"]

REFERENCE_IMG = None  # set once we load the first mask/map


# ══════════════════════════════════════════════════════════════
# Step 1 — compute the missing QPP rest-vs-fe contrast
# (mirrors the paired-t-test + FDR approach already used in
#  04_reho_analysis.py / 05_activation_contrast.py)
# ══════════════════════════════════════════════════════════════

def fdr_correct(p_map, mask, alpha=FDR_ALPHA):
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
    """Paired t-test: per-subject QPP (fe vs rest), FDR-corrected."""
    rest_imgs, fe_imgs, used = [], [], []
    for sub in subjects:
        r = QPP_RESULTS_DIR / "rest" / f"{sub}_qpp_template_MNI_ANTs.nii.gz"
        f = QPP_RESULTS_DIR / "fe"   / f"{sub}_qpp_template_MNI_ANTs.nii.gz"
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

    resample = lambda imgs: np.stack(
        [image.resample_to_img(
            nib.Nifti1Image(mid_frame(img), img.affine), ref_img,
            interpolation="continuous").get_fdata(dtype=np.float32)
         for img in imgs], axis=0)

    rest_stack = resample(rest_imgs)
    fe_stack   = resample(fe_imgs)

    mask = ref_img.get_fdata().astype(bool) if ref_img.get_fdata().dtype != bool \
        else ref_img.get_fdata().astype(bool)
    diff = fe_stack - rest_stack

    t_map = np.zeros(ref_img.shape[:3], dtype=np.float32)
    p_map = np.ones(ref_img.shape[:3], dtype=np.float32)

    diff_masked = diff[:, mask]
    n = diff_masked.shape[0]
    mean_d = diff_masked.mean(axis=0)
    std_d  = diff_masked.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_vals = np.where(std_d > 1e-10, mean_d / (std_d / np.sqrt(n)), 0.0)
    p_vals = 2 * stats.t.sf(np.abs(t_vals), df=n - 1)

    t_map[mask] = t_vals
    p_map[mask] = p_vals

    sig_mask = fdr_correct(p_map, mask)
    t_fdr = t_map.copy()
    t_fdr[~sig_mask] = 0.0

    print(f"  QPP significant voxels (FDR q<{FDR_ALPHA}): {sig_mask.sum()}")
    return nib.Nifti1Image(t_fdr, ref_img.affine)


# ══════════════════════════════════════════════════════════════
# Step 2 — load all 4 modalities, resample to common space, binarize
# ══════════════════════════════════════════════════════════════

def load_and_binarize(path, ref_img, label):
    if not Path(path).exists():
        print(f"  [{label}] MISSING: {path}")
        return None
    img = nib.load(str(path))
    img_res = image.resample_to_img(img, ref_img, interpolation="nearest")
    data = img_res.get_fdata()
    binary = (data != 0).astype(np.uint8)
    print(f"  [{label}] {binary.sum()} significant voxels")
    return binary


def combine_glm_emotions(ref_img):
    """OR across all 5 emotion contrasts = 'any emotion activation'."""
    combined = None
    for emo in EMOTIONS:
        path = GLM_DIR / f"group_{emo}_vs_noise_z_fdr.nii.gz"
        b = load_and_binarize(path, ref_img, f"GLM-{emo}")
        if b is None:
            continue
        combined = b if combined is None else np.logical_or(combined, b)
    if combined is None:
        return None
    print(f"  [GLM combined] {combined.sum()} voxels significant in >=1 emotion")
    return combined.astype(np.uint8)


# ══════════════════════════════════════════════════════════════
# Step 3 — Yeo7 network-level summary of the overlap map
# ══════════════════════════════════════════════════════════════

def get_yeo7(ref_img):
    yeo = datasets.fetch_atlas_yeo_2011(n_networks=7, thickness="thick")
    atlas_img = nib.load(yeo.maps)
    atlas_res = image.resample_to_img(atlas_img, ref_img, interpolation="nearest")
    data = atlas_res.get_fdata()
    if data.ndim == 4:
        data = data[..., 0]
    names = ["Visual", "Somatomotor", "DorsalAttention", "VentralAttention",
              "Limbic", "Frontoparietal", "Default"]
    label_map = {i + 1: n for i, n in enumerate(names)}
    return data.astype(int), label_map


def summarize_by_network(overlap_count, atlas_data, label_map, n_modalities):
    rows = []
    for roi_id, name in label_map.items():
        roi_mask = (atlas_data == roi_id)
        n_vox = roi_mask.sum()
        if n_vox == 0:
            continue
        vals = overlap_count[roi_mask]
        mean_overlap = vals.mean()
        pct_high_agreement = (vals >= n_modalities - 1).mean() * 100  # near-unanimous
        rows.append({
            "Network": name,
            "MeanOverlap": mean_overlap,
            "PctVoxels_HighAgreement": pct_high_agreement,
            "MaxOverlap": vals.max(),
            "NVoxels": int(n_vox),
        })
    df = pd.DataFrame(rows).sort_values("MeanOverlap", ascending=False)
    return df


# ══════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════

def plot_overlap_brain(overlap_img, n_modalities, out_path):
    """Categorical brain map: 0=none, up to n_modalities=all agree."""
    cmap = plt.cm.get_cmap("YlOrRd", n_modalities + 1) if hasattr(plt.cm, "get_cmap") \
        else mpl.colormaps["YlOrRd"].resampled(n_modalities + 1)

    display = plotting.plot_stat_map(
        overlap_img, display_mode="z", cut_coords=8,
        title=f"Cross-Modality Overlap (0-{n_modalities} modalities agree)",
        colorbar=True, cmap=cmap, threshold=0.5, vmax=n_modalities,
        annotate=True,
    )
    display.savefig(str(out_path), dpi=150)
    display.close()


def plot_network_bar(df, out_path, n_modalities):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d73027" if v >= (n_modalities - 1) else "#4575b4"
              for v in df["MeanOverlap"]]
    ax.barh(df["Network"], df["MeanOverlap"], color=colors, edgecolor="white")
    ax.set_xlabel(f"Mean overlap count (0-{n_modalities} modalities)")
    ax.set_title("Yeo7 Network: Cross-Modality Agreement\n"
                  "(Red = high agreement -> candidate 'cognitive'/domain-general network)")
    ax.axvline(n_modalities - 1, color="black", linestyle="--", linewidth=0.8,
               label="Near-unanimous threshold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    # Use the fALFF FDR map as the reference space (MNI 3mm) since it's
    # already in a standard resolution shared by ReHo and GLM outputs.
    reho_path = REHO_DIR / "group_rest-vs-fe_reho_t_thr.nii.gz"
    if not reho_path.exists():
        raise FileNotFoundError(
            f"{reho_path} not found — run 04_reho_analysis.py first.")
    ref_img = nib.load(str(reho_path))

    print("Loading modalities...")
    modality_masks = {}

    modality_masks["ReHo"] = load_and_binarize(reho_path, ref_img, "ReHo")
    modality_masks["fALFF"] = load_and_binarize(
        ACT_DIR / "group_fe_vs_rest_t_fdr.nii.gz", ref_img, "fALFF")
    modality_masks["GLM"] = combine_glm_emotions(ref_img)

    qpp_contrast_img = compute_qpp_contrast(ALL_SUBJECTS, ref_img)
    if qpp_contrast_img is not None:
        nib.save(qpp_contrast_img, str(OUT_DIR / "qpp_rest_vs_fe_contrast.nii.gz"))
        modality_masks["QPP"] = (qpp_contrast_img.get_fdata() != 0).astype(np.uint8)
    else:
        modality_masks["QPP"] = None

    # Drop any missing modalities but keep track of how many we actually used
    active_modalities = {k: v for k, v in modality_masks.items() if v is not None}
    n_modalities = len(active_modalities)
    print(f"\nUsing {n_modalities} modalities: {list(active_modalities.keys())}")

    if n_modalities < 2:
        raise RuntimeError("Need at least 2 modalities to compute a meaningful overlap.")

    overlap_count = np.sum(list(active_modalities.values()), axis=0).astype(np.float32)
    overlap_img = nib.Nifti1Image(overlap_count, ref_img.affine)
    nib.save(overlap_img, str(OUT_DIR / "overlap_count.nii.gz"))

    plot_overlap_brain(overlap_img, n_modalities,
                        OUT_DIR / "overlap_count_axial.png")

    # ── Yeo7 network-level summary ──────────────────────────────
    atlas_data, label_map = get_yeo7(ref_img)
    df = summarize_by_network(overlap_count, atlas_data, label_map, n_modalities)
    df.to_csv(OUT_DIR / "yeo7_network_overlap_summary.csv", index=False)
    print("\nYeo7 network overlap summary:")
    print(df.to_string(index=False))

    plot_network_bar(df, OUT_DIR / "yeo7_network_overlap_bar.png", n_modalities)

    # ── Classification: "cognitive" (domain-general) vs "emotion-specific" ──
    cognitive_networks = df[df["PctVoxels_HighAgreement"] > 10]["Network"].tolist()
    emotion_specific_networks = df[df["PctVoxels_HighAgreement"] <= 10]["Network"].tolist()

    summary = {
        "n_modalities_used": n_modalities,
        "modalities": list(active_modalities.keys()),
        "candidate_cognitive_networks": cognitive_networks,
        "candidate_emotion_specific_networks": emotion_specific_networks,
        "note": ("'Cognitive/domain-general' = networks where >10% of voxels show "
                 "near-unanimous agreement across modalities, i.e. active regardless "
                 "of which measure (ReHo/fALFF/QPP/GLM) is used. This is a descriptive "
                 "heuristic threshold, not a validated cutoff — inspect the CSV and "
                 "adjust the 10% threshold based on your actual distribution."),
    }
    with open(OUT_DIR / "overlap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nCandidate cognitive/domain-general networks: {cognitive_networks}")
    print(f"Candidate emotion-specific networks: {emotion_specific_networks}")
    print(f"\nDone. Outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()