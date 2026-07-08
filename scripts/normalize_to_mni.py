"""
Direct BOLD → MNI152 Normalization (no T1w required)
Matches the original NeuroEmo pipeline structure
(download_01.py, preprocess_02.py, extract_qpp_03.py)

Why this approach:
  T1w files for ds005700 are broken on OpenNeuro (confirmed bug).
  Direct EPI-to-MNI registration is a recognized fallback method
  when structural scans are unavailable.
"""

import os
import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img
from nilearn.datasets import load_mni152_template


TASKS = ["rest", "fe"]   # matches your original pipeline


def get_mni_target(resolution=2):
    """Load the MNI152 template at a given resolution (mm)."""
    print("Loading MNI152 template...")
    mni = load_mni152_template(resolution=resolution)
    return mni


def compute_mean_bold(clean_bold_path):
    """Average a 4D BOLD timeseries into one stable 3D image."""
    img  = nib.load(clean_bold_path)
    data = img.get_fdata()
    mean_data = data.mean(axis=3)
    mean_img_obj = nib.Nifti1Image(mean_data, img.affine, img.header)
    return mean_img_obj


def register_affine(moving_img, target_img):
    """
    Affine-register moving_img (subject's mean BOLD, native space)
    onto target_img (MNI152 template) using center-of-mass alignment.
    """
    moving_data = moving_img.get_fdata()
    target_data = target_img.get_fdata()

    def center_of_mass(data):
        total = data.sum()
        if total == 0:
            return np.array(data.shape) / 2
        coords = np.indices(data.shape)
        com = np.array([
            (coords[i] * data).sum() / total for i in range(3)
        ])
        return com

    moving_com_vox = center_of_mass(moving_data)
    target_com_vox = center_of_mass(target_data)

    moving_com_world = nib.affines.apply_affine(moving_img.affine, moving_com_vox)
    target_com_world = nib.affines.apply_affine(target_img.affine, target_com_vox)

    translation = target_com_world - moving_com_world

    new_affine = moving_img.affine.copy()
    new_affine[:3, 3] += translation

    return new_affine


def normalize_subject_qpp(sub_id, task_name, mni_template):
    """
    Full normalization for one subject's QPP result.
    Paths match your original pipeline exactly:
      data/processed/{sub_id}_task-{task}_clean_bold.nii.gz
      results/{task}/{sub_id}_qpp_template.nii.gz
    """
    clean_bold_path = f"data/processed/{sub_id}_task-{task_name}_clean_bold.nii.gz"
    qpp_path        = f"results/{task_name}/{sub_id}_qpp_template.nii.gz"
    mni_qpp_path    = f"results/{task_name}/{sub_id}_qpp_template_MNI.nii.gz"

    if os.path.exists(mni_qpp_path):
        print(f"  [{sub_id} | {task_name}] Already normalized, skipping.")
        return mni_qpp_path

    if not (os.path.exists(clean_bold_path) and os.path.exists(qpp_path)):
        print(f"  [{sub_id} | {task_name}] Missing input files:")
        print(f"    clean_bold exists: {os.path.exists(clean_bold_path)} → {clean_bold_path}")
        print(f"    qpp exists       : {os.path.exists(qpp_path)} → {qpp_path}")
        return None

    print(f"\n  [{sub_id} | {task_name}] Normalizing to MNI152...")

    # Step 1: mean BOLD image (registration target)
    mean_native = compute_mean_bold(clean_bold_path)

    # Step 2: estimate affine to MNI152
    affine_to_mni = register_affine(mean_native, mni_template)

    # Step 3+4: apply transform to QPP and resample
    qpp_img  = nib.load(qpp_path)
    qpp_data = qpp_img.get_fdata()
    n_frames = qpp_data.shape[3]
    mni_shape = mni_template.shape
    qpp_mni = np.zeros((*mni_shape, n_frames))

    for frame in range(n_frames):
        frame_img = nib.Nifti1Image(qpp_data[:, :, :, frame], affine_to_mni)
        resampled = resample_to_img(
            frame_img, mni_template,
            interpolation='continuous'
        )
        qpp_mni[:, :, :, frame] = resampled.get_fdata()

    # Step 5: save
    os.makedirs(f"results/{task_name}", exist_ok=True)
    mni_img = nib.Nifti1Image(qpp_mni, mni_template.affine)
    nib.save(mni_img, mni_qpp_path)
    print(f"  [{sub_id} | {task_name}] Saved → {mni_qpp_path}")

    return mni_qpp_path


if __name__ == "__main__":
    mni_template = get_mni_target(resolution=2)

    # Process all subjects 1-40 (will skip any missing/already-done ones)
    for i in range(1, 41):
        sub_id = f"sub-{i:02d}"
        for task_name in TASKS:
            try:
                normalize_subject_qpp(sub_id, task_name, mni_template)
            except Exception as e:
                print(f"  ERROR on {sub_id} | {task_name}: {e}")
                continue