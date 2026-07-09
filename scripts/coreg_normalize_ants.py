"""
Step 4 + 5 of preprocessing — the two steps we skipped earlier.

Step 4: Coregistration  — align mean BOLD (native space) to T1w (native space)
Step 5: Normalization   — warp T1w (native space) to MNI152 (standard space)

Then compose both transforms and apply to your EXISTING QPP results
(no need to redo QPP extraction — those results stay valid).
"""

import os
import numpy as np
import nibabel as nib
import ants


TASKS = ["rest", "fe"]

PROCESSED_DIR = "data/processed"
RESULTS_DIR   = "results"


def get_mni_template():
    """
    Load a standard MNI152 template using nilearn, convert for ANTsPy.
    """
    from nilearn.datasets import load_mni152_template
    nib_template = load_mni152_template(resolution=2)

    tmp_path = "data/mni152_template_tmp.nii.gz"
    nib.save(nib_template, tmp_path)
    mni_img = ants.image_read(tmp_path)
    return mni_img, nib_template.affine


def compute_mean_bold(clean_bold_path):
    """Average the 4D BOLD into a stable 3D image (native space)."""
    img  = nib.load(clean_bold_path)
    data = img.get_fdata()
    mean_data = data.mean(axis=3)
    mean_img = nib.Nifti1Image(mean_data, img.affine, img.header)

    tmp_path = clean_bold_path.replace(".nii.gz", "_meanBOLD_tmp.nii.gz")
    nib.save(mean_img, tmp_path)
    return tmp_path


def coreg_and_normalize_subject(sub_id, task_name, mni_ants, mni_affine):
    """
    Full Step 4 + Step 5 for one subject/task.
    """
    clean_bold_path = f"{PROCESSED_DIR}/{sub_id}_task-{task_name}_clean_bold.nii.gz"
    t1w_path        = f"data/raw/{sub_id}/anat/{sub_id}_T1w.nii.gz"
    qpp_path        = f"{RESULTS_DIR}/{task_name}/{sub_id}_qpp_template.nii.gz"
    mni_qpp_path    = f"{RESULTS_DIR}/{task_name}/{sub_id}_qpp_template_MNI_ANTs.nii.gz"

    print(f"\n  [{sub_id} | {task_name}] Coreg + Normalize...")

    if os.path.exists(mni_qpp_path):
        print(f"    Already done, skipping.")
        return mni_qpp_path

    if not os.path.exists(clean_bold_path):
        print(f"    Missing clean BOLD: {clean_bold_path}")
        return None
    if not os.path.exists(t1w_path):
        print(f"    Missing T1w: {t1w_path}")
        return None
    if not os.path.exists(qpp_path):
        print(f"    Missing QPP: {qpp_path}")
        return None

    mean_bold_path = compute_mean_bold(clean_bold_path)
    bold_ants = ants.image_read(mean_bold_path)
    t1w_ants  = ants.image_read(t1w_path)

    print(f"    Step 4: Coregistering BOLD to T1w (rigid)...")
    coreg = ants.registration(
        fixed=t1w_ants,
        moving=bold_ants,
        type_of_transform="Rigid"
    )
    bold_to_t1w_transform = coreg["fwdtransforms"]

    print(f"    Step 5: Normalizing T1w to MNI152 (SyN nonlinear)...")
    norm = ants.registration(
        fixed=mni_ants,
        moving=t1w_ants,
        type_of_transform="SyN"
    )
    t1w_to_mni_transform = norm["fwdtransforms"]

    print(f"    Applying composed transform to QPP (10 frames)...")
    qpp_img  = nib.load(qpp_path)
    qpp_data = qpp_img.get_fdata()
    n_frames = qpp_data.shape[3]

    mni_frames = []
    for frame in range(n_frames):
        frame_data = qpp_data[:, :, :, frame]
        frame_nib  = nib.Nifti1Image(frame_data, qpp_img.affine)

        tmp_frame_path = "data/qpp_frame_tmp.nii.gz"
        nib.save(frame_nib, tmp_frame_path)
        frame_ants = ants.image_read(tmp_frame_path)

        warped = ants.apply_transforms(
            fixed=mni_ants,
            moving=frame_ants,
            transformlist=t1w_to_mni_transform + bold_to_t1w_transform,
            interpolator="linear"
        )
        mni_frames.append(warped.numpy())

    mni_data = np.stack(mni_frames, axis=3)
    mni_img  = nib.Nifti1Image(mni_data, mni_affine)

    os.makedirs(f"{RESULTS_DIR}/{task_name}", exist_ok=True)
    nib.save(mni_img, mni_qpp_path)
    print(f"    Saved -> {mni_qpp_path}")

    if os.path.exists(mean_bold_path):
        os.remove(mean_bold_path)

    return mni_qpp_path


if __name__== "__main__":
    print("Loading MNI152 template...")
    mni_ants, mni_affine = get_mni_template()

    ALL_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 41)]

    success = []
    failed  = []

    for sub_id in ALL_SUBJECTS:
        for task_name in TASKS:
            try:
                result = coreg_and_normalize_subject(sub_id, task_name, mni_ants, mni_affine)
                if result is not None:
                    success.append(f"{sub_id}_{task_name}")
                else:
                    failed.append(f"{sub_id}_{task_name}")
            except Exception as e:
                print(f"  ERROR on {sub_id} | {task_name}: {e}")
                import traceback
                traceback.print_exc()
                failed.append(f"{sub_id}_{task_name}")
                continue

    print("\n" + "="*60)
    print("COREGISTRATION + NORMALIZATION COMPLETE")
    print("="*60)
    print(f"Successful: {len(success)} / {len(ALL_SUBJECTS)*len(TASKS)}")
    print(f"Failed    : {len(failed)}")
    if failed:
        print(f"Failed list: {failed}")