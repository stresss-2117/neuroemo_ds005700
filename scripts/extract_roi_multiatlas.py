"""
Multi-Atlas ROI Extraction for QPP results.

Design goal: extract the SAME QPP data through DIFFERENT atlas lenses,
so you can compare AAL vs Schaefer vs Yeo (or run just one at a time).

Output format matches your professor's extract_results_wrapper():
  DataFrame with columns ['Atlas', 'ROI', 'Weights']
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets
from nilearn.image import resample_to_img


# ══════════════════════════════════════════════════════════════
# ATLAS REGISTRY — add/remove atlases here, nothing else changes
# ══════════════════════════════════════════════════════════════

def load_aal():
    """AAL116 — anatomical (gyral) atlas, cortex+subcortex+cerebellum."""
    aal = datasets.fetch_atlas_aal()
    atlas_img = nib.load(aal.maps)
    labels = aal.labels          # list of region names
    indices = aal.indices        # list of region ID numbers (as strings)
    label_map = {int(idx): name for idx, name in zip(indices, labels)}
    return atlas_img, label_map


def load_schaefer(n_rois=400, yeo_networks=7):
    """Schaefer — functional (rs-fMRI based), cortex only."""
    schaefer = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois, yeo_networks=yeo_networks, resolution_mm=2
    )
    atlas_img = nib.load(schaefer.maps)
    labels = [l.decode() if isinstance(l, bytes) else l for l in schaefer.labels]
    label_map = {i + 1: name for i, name in enumerate(labels)}
    return atlas_img, label_map


def load_yeo(n_networks=7):
    """Yeo — network-level functional atlas, cortex only."""
    yeo = datasets.fetch_atlas_yeo_2011(n_networks=n_networks, thickness="thick")
    atlas_img = nib.load(yeo.maps)   # or yeo['maps']
    network_names_7 = [
        "Visual", "Somatomotor", "DorsalAttention", "VentralAttention",
        "Limbic", "Frontoparietal", "Default"
    ]
    network_names_17 = [f"Network{i}" for i in range(1, 18)]
    names = network_names_7 if n_networks == 7 else network_names_17
    label_map = {i + 1: name for i, name in enumerate(names)}
    return atlas_img, label_map


# Registry: name -> loader function
ATLAS_REGISTRY = {
    "aal":      load_aal,
    "schaefer": load_schaefer,
    "yeo7":     lambda: load_yeo(7),
    "yeo17":    lambda: load_yeo(17),
}


# ══════════════════════════════════════════════════════════════
# CORE EXTRACTION — works identically for ANY atlas
# ══════════════════════════════════════════════════════════════

def extract_roi_values(qpp_img, atlas_img, label_map, frame=None):
    qpp_data = qpp_img.get_fdata()

    if qpp_data.ndim == 4:
        n_frames = qpp_data.shape[3]
        use_frame = frame if frame is not None else n_frames // 2
        qpp_3d = qpp_data[:, :, :, use_frame]
    else:
        qpp_3d = qpp_data

    qpp_3d_img = nib.Nifti1Image(qpp_3d, qpp_img.affine)

    atlas_resampled = resample_to_img(
        atlas_img, qpp_3d_img, interpolation='nearest'
    )
    atlas_data = atlas_resampled.get_fdata().astype(int)
    if atlas_data.ndim == 4:
        atlas_data = atlas_data[..., 0]   # squeeze Yeo's trailing singleton dim

    rows = []
    for roi_id, roi_name in label_map.items():
        roi_mask = (atlas_data == roi_id)
        if roi_mask.sum() == 0:
            continue
        mean_val = qpp_3d[roi_mask].mean()
        rows.append({"ROI": roi_name, "Weights": mean_val})

    return pd.DataFrame(rows)


def extract_multiatlas(qpp_path, atlas_names, frame=None):
    """
    Extract ROI values for the SAME QPP file across MULTIPLE atlases.

    qpp_path    : path to a .nii.gz QPP file (native, MNI-affine, or MNI-ANTs)
    atlas_names : list like ["schaefer"], ["aal","schaefer"], or all keys

    Returns: DataFrame with columns ['Atlas', 'ROI', 'Weights']
             — ready for your professor's extract_results_wrapper()
    """
    qpp_img = nib.load(qpp_path)
    all_rows = []

    for atlas_name in atlas_names:
        if atlas_name not in ATLAS_REGISTRY:
            print(f"  Unknown atlas '{atlas_name}', skipping. "
                  f"Available: {list(ATLAS_REGISTRY.keys())}")
            continue

        print(f"  Loading atlas: {atlas_name}...")
        atlas_img, label_map = ATLAS_REGISTRY[atlas_name]()

        print(f"  Extracting ROI values ({len(label_map)} regions)...")
        df = extract_roi_values(qpp_img, atlas_img, label_map, frame=frame)
        df["Atlas"] = atlas_name

        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame(columns=["Atlas", "ROI", "Weights"])

    result = pd.concat(all_rows, ignore_index=True)
    return result[["Atlas", "ROI", "Weights"]]


# ══════════════════════════════════════════════════════════════
# DRIVER — extract for group-average QPP, one or many atlases
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    os.makedirs("results/roi_tables", exist_ok=True)

    ALL_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 41)]
    TASKS = ["rest", "fe"]

    # ── CONFIGURE YOUR ATLAS MIX HERE ─────────────────────────────
    ATLASES_TO_USE = ["aal", "schaefer", "yeo7"]   # mix of 3
    # ATLASES_TO_USE = ["schaefer"]                 # just one
    # ATLASES_TO_USE = list(ATLAS_REGISTRY.keys())  # everything

    def compute_group_average_qpp(task_name, subject_ids):
        """Average all subjects' MNI-ANTs QPP (middle frame) for one task."""
        all_data = []
        affine = None
        used = []

        for sub_id in subject_ids:
            path = f"results/{task_name}/{sub_id}_qpp_template_MNI_ANTs.nii.gz"
            if not os.path.exists(path):
                continue
            img = nib.load(path)
            data = img.get_fdata()
            mid = data.shape[3] // 2
            all_data.append(data[:, :, :, mid])
            used.append(sub_id)
            if affine is None:
                affine = img.affine

        if not all_data:
            return None, []

        group_data = np.mean(all_data, axis=0)
        group_img = nib.Nifti1Image(group_data, affine)
        return group_img, used

    # ── Run for each task ──────────────────────────────────────────
    all_results = {}

    for task_name in TASKS:
        print(f"\n{'='*60}")
        print(f"TASK: {task_name}")
        print(f"{'='*60}")

        group_img, used = compute_group_average_qpp(task_name, ALL_SUBJECTS)
        if group_img is None:
            print(f"  No subjects found for {task_name}, skipping.")
            continue

        print(f"  Group average from {len(used)} subjects")

        tmp_path = f"results/{task_name}/group_average_MNI_ANTs.nii.gz"
        nib.save(group_img, tmp_path)

        all_rows = []
        for atlas_name in ATLASES_TO_USE:
            if atlas_name not in ATLAS_REGISTRY:
                print(f"  Unknown atlas '{atlas_name}', skipping.")
                continue
            print(f"\n  Loading atlas: {atlas_name}...")
            atlas_img, label_map = ATLAS_REGISTRY[atlas_name]()
            print(f"  Extracting ROI values ({len(label_map)} regions)...")
            df = extract_roi_values(group_img, atlas_img, label_map)
            df["Atlas"] = atlas_name
            df["Task"] = task_name
            all_rows.append(df)

        df_task = pd.concat(all_rows, ignore_index=True)
        df_task = df_task[df_task["ROI"] != "Background"]   # clean up

        out_path = f"results/roi_tables/group_{task_name}_multiatlas.csv"
        df_task.to_csv(out_path, index=False)
        print(f"\n  Saved -> {out_path}  ({len(df_task)} rows)")

        all_results[task_name] = df_task

    # ── Combine rest + emotion into one master table ────────────────
    if len(all_results) == 2:
        df_combined = pd.concat(all_results.values(), ignore_index=True)
        combined_path = "results/roi_tables/group_all_tasks_multiatlas.csv"
        df_combined.to_csv(combined_path, index=False)
        print(f"\nCombined table saved -> {combined_path}")
        print(f"Total rows: {len(df_combined)}")

    print("\nDone! ROI tables ready for plotting.")