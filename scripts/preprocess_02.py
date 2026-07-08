import numpy as np
import nibabel as nib
import os
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from nilearn.image import smooth_img
from nilearn.masking import compute_epi_mask

def preprocess_subject(sub_id, task, tr=None):
    """
    Full preprocessing pipeline for one subject + one task.

    sub_id : string e.g. "sub-01"
    task   : "rest" or "fe" (emotion)
    tr     : repetition time in seconds (auto-detected if None)

    Returns: clean_bold (numpy array), mask (numpy array), tr (float)
    """

    print(f"\n  [{sub_id} | task-{task}] Starting preprocessing...")

    # ── Paths ──────────────────────────────────────────────────
    bold_path = f"data/raw/{sub_id}/func/{sub_id}_task-{task}_bold.nii.gz"
    out_path  = f"data/processed/{sub_id}_task-{task}_clean_bold.nii.gz"
    mask_path = f"data/processed/{sub_id}_task-{task}_mask.npy"

    # Skip if already preprocessed
    if os.path.exists(out_path) and os.path.exists(mask_path):
        print(f"  [{sub_id} | task-{task}] Already preprocessed! Loading...")
        img        = nib.load(out_path)
        clean_bold = img.get_fdata()
        mask       = np.load(mask_path)
        return clean_bold, mask, tr or 2.0

    # Check file exists
    if not os.path.exists(bold_path):
        print(f"  ERROR: {bold_path} not found!")
        return None, None, None

    # ── Load data ───────────────────────────────────────────────
    print(f"  [{sub_id}] Loading BOLD...")
    img  = nib.load(bold_path)
    bold = img.get_fdata()
    print(f"  [{sub_id}] Shape: {bold.shape}")

    # Auto-detect TR from header if not provided
    if tr is None:
        tr = float(img.header.get_zooms()[3])
        if tr < 0.1 or tr > 20:
            tr = 2.0  # fallback
    print(f"  [{sub_id}] TR = {tr}s")

    X, Y, Z, T = bold.shape

    # ── Step 1: Slice Timing Correction ────────────────────────
    print(f"  [{sub_id}] Step 1: Slice timing correction...")
    n_slices      = Z
    slice_times   = np.linspace(0, tr * (1 - 1/n_slices), n_slices)
    reference_time = tr / 2.0
    original_times = np.arange(T) * tr
    bold_stc       = np.zeros_like(bold)

    for z in range(n_slices):
        shift         = slice_times[z] - reference_time
        shifted_times = original_times + shift
        for x in range(X):
            for y in range(Y):
                ts = bold[x, y, z, :]
                if ts.std() > 0:
                    f = interp1d(shifted_times, ts, kind='linear',
                                 bounds_error=False,
                                 fill_value=(ts[0], ts[-1]))
                    bold_stc[x, y, z, :] = f(original_times)
                else:
                    bold_stc[x, y, z, :] = ts

    # ── Step 2: Brain Extraction ────────────────────────────────
    print(f"  [{sub_id}] Step 2: Brain extraction...")
    bold_img = nib.Nifti1Image(bold_stc, img.affine, img.header)
    mask_img = compute_epi_mask(bold_img)
    mask     = mask_img.get_fdata()
    bold_masked = bold_stc * mask[:, :, :, np.newaxis]
    brain_voxels = np.where(mask > 0)
    print(f"  [{sub_id}] Brain voxels: {len(brain_voxels[0]):,}")

    # ── Step 3: Spatial Smoothing ───────────────────────────────
    print(f"  [{sub_id}] Step 3: Spatial smoothing (6mm FWHM)...")
    bold_masked_img = nib.Nifti1Image(bold_masked, img.affine, img.header)
    bold_smooth_img = smooth_img(bold_masked_img, fwhm=6)
    bold_smooth     = bold_smooth_img.get_fdata()

    # ── Step 4: Bandpass Filter ─────────────────────────────────
    print(f"  [{sub_id}] Step 4: Bandpass filter (0.01-0.1 Hz)...")
    nyquist    = 1.0 / (2.0 * tr)
    high_pass  = 0.01
    low_pass   = 0.10
    b, a       = butter(N=2,
                        Wn=[high_pass/nyquist, low_pass/nyquist],
                        btype='band')
    bold_filt  = np.zeros_like(bold_smooth)

    for bv in zip(*brain_voxels):
        x, y, z = bv
        ts = bold_smooth[x, y, z, :]
        if ts.std() > 0:
            bold_filt[x, y, z, :] = filtfilt(b, a, ts)

    # ── Step 5: Confound Regression ─────────────────────────────
    print(f"  [{sub_id}] Step 5: Confound regression...")
    brain_ts     = bold_filt[brain_voxels[0],
                              brain_voxels[1],
                              brain_voxels[2], :]
    global_sig   = brain_ts.mean(axis=0)
    mean_vol     = bold_filt.mean(axis=3)
    csf_thresh   = np.percentile(mean_vol[mean_vol > 0], 95)
    wm_thresh    = np.percentile(mean_vol[mean_vol > 0], 70)
    csf_mask     = (mean_vol > csf_thresh) & (mask > 0)
    wm_mask      = (mean_vol > wm_thresh)  & (mask > 0)
    csf_sig      = bold_filt[csf_mask].mean(axis=0)
    wm_sig       = bold_filt[wm_mask].mean(axis=0)
    linear_trend = np.linspace(-1, 1, T)
    confounds    = np.column_stack([
        global_sig, csf_sig, wm_sig,
        linear_trend, np.ones(T)
    ])

    bold_clean = np.zeros_like(bold_filt)
    for x, y, z in zip(*brain_voxels):
        ts    = bold_filt[x, y, z, :]
        betas, _, _, _ = np.linalg.lstsq(confounds, ts, rcond=None)
        bold_clean[x, y, z, :] = ts - confounds @ betas

    # ── Step 6: Z-score ─────────────────────────────────────────
    print(f"  [{sub_id}] Step 6: Z-scoring...")
    for x, y, z in zip(*brain_voxels):
        ts  = bold_clean[x, y, z, :]
        std = ts.std()
        if std > 0:
            bold_clean[x, y, z, :] = (ts - ts.mean()) / std

    # ── Save ────────────────────────────────────────────────────
    os.makedirs("data/processed", exist_ok=True)
    clean_img = nib.Nifti1Image(bold_clean, img.affine, img.header)
    nib.save(clean_img, out_path)
    np.save(mask_path, mask)
    print(f"  [{sub_id} | task-{task}] Preprocessing done! Saved → {out_path}")

    return bold_clean, mask, tr