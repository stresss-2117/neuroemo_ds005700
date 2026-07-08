"""
QPP Brain Plotting — adapted from Winson's plot_fig() code
Modified ONLY in the data-loading and driver sections.
plot_fig() itself is UNCHANGED from the original.
"""

import pandas as pd
import numpy as np
import os

import nibabel as nb
from nilearn.image import load_img, math_img, index_img
from nilearn.datasets import load_mni152_template, fetch_surf_fsaverage
from nilearn import plotting, surface, datasets

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps

template = load_mni152_template(resolution=1)


# ══════════════════════════════════════════════════════════════
# plot_fig() — EXACT COPY from Winson's notebook, UNCHANGED
# ══════════════════════════════════════════════════════════════
def plot_fig(tmap, nrow, ncol, figsize, annotate=False, save=False, close=False,
             display_mode='x', labelsize=12, orientation='vertical',
             cbar_label='ReHo Difference', plot_save_path='figure.png'):
    """
    Plot multiple slices of a 3D statistical map.
    (identical to the version your professor provided)
    """

    if isinstance(tmap, str):
        img = nb.load(tmap)
    else:
        img = tmap

    cut_coords = {
        'x': np.linspace(-60, 60, ncol * nrow, dtype=int).tolist(),
        'y': np.linspace(-100, 70, ncol * nrow, dtype=int).tolist(),
        'z': np.linspace(-62, 72, ncol * nrow, dtype=int).tolist()
    }
    print(f'Display mode: {display_mode}, \nCoordinates: {cut_coords[display_mode]}')
    counter = 0

    fig, axes = plt.subplots(nrow, ncol+1, figsize=figsize, gridspec_kw={'width_ratios': [1]*ncol + [0.05]})
    gs = axes[0, ncol].get_gridspec()
    for ax in axes[0:, -1]:
        ax.remove()
    ax_colorbar = fig.add_subplot(gs[0:, -1])

    for row in range(0, nrow):
        for col in range(0, ncol):
            plotting.plot_stat_map(img, title='', black_bg=True, bg_img=template,
                                   axes=axes[row, col], annotate=annotate,
                                   display_mode=display_mode, symmetric_cbar=True,
                                   colorbar=False, cut_coords=[cut_coords[display_mode][counter]])
            counter += 1

    font_properties = {'family': 'serif', 'size': labelsize}
    cmap = cm.get_cmap('cold_hot')
    fdata = img.get_fdata()
    fdata_max = np.abs(fdata).max()
    vmax_f = fdata_max + (fdata_max * 0.05)
    norm = mpl.colors.Normalize(vmin=-vmax_f, vmax=vmax_f)

    cb1 = mpl.colorbar.ColorbarBase(ax_colorbar, cmap=cmap, norm=norm, orientation=orientation)
    cb1.set_label(cbar_label)
    cb1.ax.tick_params(labelsize=labelsize)
    cb1.ax.set_yticklabels(cb1.ax.get_yticklabels(), **font_properties)

    if save:
        plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    if close:
        plt.close()

    return fig, axes


# ══════════════════════════════════════════════════════════════
# NEW — loads YOUR QPP data (replaces extract_results_wrapper etc.)
# ══════════════════════════════════════════════════════════════

def load_subject_qpp(sub_id, task_name, frame=None):
    """
    Load one subject's MNI-normalized QPP template and extract one 3D frame.

    sub_id    : e.g. "sub-01"
    task_name : "rest" or "fe"
    frame     : which of the 10 QPP timeframes to use (default: middle frame)

    Returns: nibabel Nifti1Image (3D) ready for plot_fig(), or None if missing
    """
    mni_qpp_path = f"results/{task_name}/{sub_id}_qpp_template_MNI.nii.gz"

    if not os.path.exists(mni_qpp_path):
        return None

    qpp_img = nb.load(mni_qpp_path)
    n_frames = qpp_img.shape[3]

    if frame is None:
        frame = n_frames // 2

    return index_img(qpp_img, frame)


def compute_group_average(task_name, subject_ids, frame=None):
    """
    Average the MNI-space QPP templates across all given subjects
    for one task. This is the "combined plot" step.

    task_name   : "rest" or "fe"
    subject_ids : list like ["sub-01", "sub-02", ...]
    frame       : which QPP timeframe to average (default: middle frame)

    Returns: nibabel Nifti1Image (3D) — the group-average brain map
    """
    all_data = []
    affine = None
    used_subjects = []

    for sub_id in subject_ids:
        mni_qpp_path = f"results/{task_name}/{sub_id}_qpp_template_MNI.nii.gz"
        if not os.path.exists(mni_qpp_path):
            print(f"  Skipping {sub_id} ({task_name}) — file not found")
            continue

        qpp_img = nb.load(mni_qpp_path)
        n_frames = qpp_img.shape[3]
        use_frame = frame if frame is not None else n_frames // 2

        frame_data = qpp_img.get_fdata()[:, :, :, use_frame]
        all_data.append(frame_data)
        used_subjects.append(sub_id)

        if affine is None:
            affine = qpp_img.affine

    if len(all_data) == 0:
        print(f"  No subjects found for task '{task_name}'!")
        return None, []

    print(f"  Averaging {len(all_data)} subjects for task '{task_name}': {used_subjects}")
    group_data = np.mean(all_data, axis=0)
    group_img = nb.Nifti1Image(group_data, affine)

    return group_img, used_subjects


# ══════════════════════════════════════════════════════════════
# Driver code — replaces the bottom section of the original notebook
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    os.makedirs("results/figures", exist_ok=True)

    ALL_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 41)]
    TASKS = ["rest", "fe"]

    # ── PART 1: Single subject plot (sub-01) ─────────────────────
    print("\n" + "="*60)
    print("PART 1: Single subject brain plots")
    print("="*60)

    for task_name in TASKS:
        print(f"\n[sub-01 | {task_name}]")
        tmap = load_subject_qpp("sub-01", task_name)

        if tmap is None:
            print(f"  No QPP data found for sub-01 / {task_name}, skipping.")
            continue

        plot_fig(
            tmap, nrow=2, ncol=4, figsize=(16, 8),
            display_mode='z',
            cbar_label=f'sub-01 {task_name} QPP (Z-score)',
            save=True, close=True,
            plot_save_path=f"results/figures/sub-01_{task_name}_qpp.png"
        )
        print(f"  Saved → results/figures/sub-01_{task_name}_qpp.png")

    # ── PART 2: Group-average plot (all available subjects) ──────
    print("\n" + "="*60)
    print("PART 2: Group-average brain plots")
    print("="*60)

    group_imgs = {}
    for task_name in TASKS:
        print(f"\n[GROUP AVERAGE | {task_name}]")
        group_img, used = compute_group_average(task_name, ALL_SUBJECTS)

        if group_img is None:
            continue

        group_imgs[task_name] = group_img

        plot_fig(
            group_img, nrow=2, ncol=4, figsize=(16, 8),
            display_mode='z',
            cbar_label=f'Group Average {task_name.upper()} QPP (n={len(used)})',
            save=True, close=True,
            plot_save_path=f"results/figures/group_average_{task_name}_qpp.png"
        )
        print(f"  Saved → results/figures/group_average_{task_name}_qpp.png")

    # ── PART 3: Rest vs Emotion difference map ────────────────────
    if "rest" in group_imgs and "fe" in group_imgs:
        print("\n" + "="*60)
        print("PART 3: Rest vs Emotion difference map")
        print("="*60)

        diff_data = group_imgs["fe"].get_fdata() - group_imgs["rest"].get_fdata()
        diff_img = nb.Nifti1Image(diff_data, group_imgs["rest"].affine)

        plot_fig(
            diff_img, nrow=2, ncol=4, figsize=(16, 8),
            display_mode='z',
            cbar_label='Emotion minus Rest QPP difference',
            save=True, close=True,
            plot_save_path="results/figures/group_diff_emotion_minus_rest.png"
        )
        print("  Saved → results/figures/group_diff_emotion_minus_rest.png")

    print("\nAll plots complete! Check results/figures/")