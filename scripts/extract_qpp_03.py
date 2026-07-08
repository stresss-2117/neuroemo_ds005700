"""
Script 3: QPP Extraction for NeuroEmo dataset
Runs QPP on ONE subject, ONE task
Called by 05_run_pipeline.py in a loop
"""

import numpy as np
import nibabel as nib
import os
from scipy.signal import find_peaks


def compute_correlation(template, data_brain, window_len):
    """Slide template across timeseries, return correlation at each position."""
    n_vox, T  = data_brain.shape
    n_windows = T - window_len + 1
    corr      = np.zeros(n_windows)
    t_flat    = template.flatten()
    t_norm    = t_flat - t_flat.mean()
    t_std     = t_norm.std()
    if t_std == 0:
        return corr
    for t in range(n_windows):
        segment = data_brain[:, t : t + window_len]
        s_flat  = segment.flatten()
        s_norm  = s_flat - s_flat.mean()
        s_std   = s_norm.std()
        if s_std > 0:
            corr[t] = np.dot(t_norm, s_norm) / (len(t_norm) * t_std * s_std)
    return corr


def extract_qpp(data_brain, window_len, n_iter=10,
                peak_thresh=0.2, n_starts=3):
    """
    Iterative QPP extraction with multiple random starts.
    Returns best template, peak times, correlation timecourse.
    """
    n_vox, T  = data_brain.shape
    rng        = np.random.default_rng(42)
    best_peaks = np.array([])
    best_template = None
    best_corr  = np.zeros(T - window_len + 1)
    best_score = -1

    for start_num in range(n_starts):
        start    = rng.integers(0, T - window_len)
        template = data_brain[:, start : start + window_len].copy()
        thresh   = peak_thresh

        for iteration in range(n_iter):
            corr   = compute_correlation(template, data_brain, window_len)
            peaks, _ = find_peaks(corr, height=thresh,
                                  distance=window_len // 2)
            if len(peaks) < 3:
                thresh *= 0.85
                if thresh < 0.05:
                    break
                continue

            valid_peaks  = [p for p in peaks if p + window_len <= T]
            windows      = np.array([data_brain[:, p : p + window_len]
                                     for p in valid_peaks])
            new_template = windows.mean(axis=0)
            similarity   = np.corrcoef(
                template.flatten(), new_template.flatten()
            )[0, 1]
            template = new_template

            if similarity > 0.999:
                break

        if len(peaks) > 0:
            score = len(peaks) * corr[peaks].mean()
            if score > best_score:
                best_score    = score
                best_peaks    = peaks
                best_template = template
                best_corr     = corr

    return best_template, best_peaks, best_corr


def run_qpp_for_subject(sub_id, task, tr=2.0,
                         window_sec=20, n_iter=10):
    """
    Load preprocessed data, run QPP, save results.

    sub_id : e.g. "sub-01"
    task   : "rest" or "fe"
    Returns: dict with QPP results or None if failed
    """
    print(f"\n  [{sub_id} | task-{task}] QPP extraction...")

    # ── Paths ───────────────────────────────────────────────────
    bold_path = f"data/processed/{sub_id}_task-{task}_clean_bold.nii.gz"
    mask_path = f"data/processed/{sub_id}_task-{task}_mask.npy"
    qpp_out   = f"results/{task}/{sub_id}_qpp_template.nii.gz"
    peaks_out = f"results/{task}/{sub_id}_qpp_peaks.npy"
    corr_out  = f"results/{task}/{sub_id}_qpp_corr.npy"
    info_out  = f"results/{task}/{sub_id}_qpp_info.npy"

    # Skip if already done
    if os.path.exists(qpp_out) and os.path.exists(peaks_out):
        print(f"  [{sub_id} | task-{task}] QPP already extracted! Skipping.")
        peaks = np.load(peaks_out)
        info  = np.load(info_out, allow_pickle=True).item()
        return info

    # Check preprocessed data exists
    if not os.path.exists(bold_path):
        print(f"  ERROR: {bold_path} not found! Run preprocessing first.")
        return None

    # ── Load ────────────────────────────────────────────────────
    img       = nib.load(bold_path)
    bold      = img.get_fdata()
    mask      = np.load(mask_path)
    X, Y, Z, T = bold.shape

    # Flatten to brain voxels only
    data          = bold.reshape(-1, T)
    mask_flat     = mask.reshape(-1)
    brain_idx     = np.where(mask_flat > 0)[0]
    data_brain    = data[brain_idx, :]

    print(f"  [{sub_id}] Brain voxels: {len(brain_idx):,} | Timepoints: {T}")

    # ── QPP ─────────────────────────────────────────────────────
    window_len = int(window_sec / tr)
    template, peak_times, corr = extract_qpp(
        data_brain, window_len, n_iter=n_iter
    )

    if template is None or len(peak_times) == 0:
        print(f"  [{sub_id} | task-{task}] WARNING: No QPP found!")
        return None

    # ── Reshape template to brain volume ────────────────────────
    qpp_volume = np.zeros((X, Y, Z, window_len))
    qpp_volume.reshape(-1, window_len)[brain_idx, :] = template

    # ── Save ────────────────────────────────────────────────────
    os.makedirs(f"results/{task}", exist_ok=True)
    qpp_img = nib.Nifti1Image(qpp_volume, img.affine, img.header)
    nib.save(qpp_img, qpp_out)
    np.save(peaks_out, peak_times)
    np.save(corr_out,  corr)

    # Save summary info
    T_total  = T * tr
    interval = T_total / len(peak_times) if len(peak_times) > 0 else 0
    info = {
        "sub_id"       : sub_id,
        "task"         : task,
        "n_peaks"      : len(peak_times),
        "mean_corr"    : float(corr[peak_times].mean()),
        "avg_interval" : float(interval),
        "peak_times_s" : peak_times * tr,
        "window_sec"   : window_sec,
        "tr"           : tr,
        "n_timepoints" : T,
        "n_brain_vox"  : len(brain_idx),
    }
    np.save(info_out, info)

    print(f"  [{sub_id} | task-{task}] QPP done!")
    print(f"    Occurrences : {len(peak_times)}")
    print(f"    Mean corr   : {info['mean_corr']:.3f}")
    print(f"    Avg interval: {interval:.1f}s")

    return info