"""
ROI-parcellated brain plots — uses your professor's EXACT
vals2atlas() and plot_fig() functions, unchanged.

Only the atlas-path/csv setup is adapted to use nilearn's
built-in fetchers instead of manually-downloaded atlas files.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nb
from nilearn import datasets
from nilearn.image import load_img
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

template = datasets.load_mni152_template(resolution=1)


# ══════════════════════════════════════════════════════════════
# plot_fig() — EXACT COPY, unchanged from your professor's code
# ══════════════════════════════════════════════════════════════
def plot_fig(tmap, nrow, ncol, figsize, annotate=False, save=False, close=False,
             display_mode='x', labelsize=12, orientation='vertical',
             cbar_label='ReHo Difference', plot_save_path='figure.png'):

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

    from nilearn import plotting
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
    cmap = mpl.colormaps['cold_hot']
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
# vals2atlas() — SAME LOGIC as your professor's version,
# adapted to use nilearn's atlas objects directly (no CSV files
# on disk needed, since nilearn already gives us labels in-memory)
# ══════════════════════════════════════════════════════════════

def vals2atlas_nilearn(df, selected_atlas, atlas_img, label_map, stat='Weights'):
    print(f'\t{selected_atlas}...', end='')

    atlas_data = atlas_img.get_fdata().copy()
    if atlas_data.ndim == 4:
        atlas_data = atlas_data[..., 0]   # squeeze Yeo's trailing singleton dim

    results_selected_atlas = df[df['Atlas'] == selected_atlas]
    roi_to_val = dict(zip(results_selected_atlas['ROI'], results_selected_atlas[stat]))

    out_data = np.zeros_like(atlas_data)
    for roi_id, roi_name in label_map.items():
        if roi_name in roi_to_val:
            out_data[atlas_data == roi_id] = roi_to_val[roi_name]

    tmap = nb.Nifti1Image(out_data, header=atlas_img.header, affine=atlas_img.affine)
    print(' done')
    return tmap


# ══════════════════════════════════════════════════════════════
# Load atlases (same registry as extract_roi_multiatlas.py)
# ══════════════════════════════════════════════════════════════

def get_atlas(atlas_name):
    if atlas_name == "aal":
        aal = datasets.fetch_atlas_aal()
        atlas_img = nb.load(aal.maps)
        label_map = {int(idx): name for idx, name in zip(aal.indices, aal.labels)}
    elif atlas_name == "schaefer":
        schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=2)
        atlas_img = nb.load(schaefer.maps)
        labels = [l.decode() if isinstance(l, bytes) else l for l in schaefer.labels]
        label_map = {i + 1: name for i, name in enumerate(labels)}
    elif atlas_name == "yeo7":
        yeo = datasets.fetch_atlas_yeo_2011(n_networks=7, thickness="thick")
        atlas_img = nb.load(yeo.maps)
        names = ["Visual","Somatomotor","DorsalAttention","VentralAttention",
                 "Limbic","Frontoparietal","Default"]
        label_map = {i + 1: name for i, name in enumerate(names)}
    else:
        raise ValueError(f"Unknown atlas: {atlas_name}")
    return atlas_img, label_map


# ══════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    os.makedirs("results/figures/roi_plots", exist_ok=True)

    TASKS = ["rest", "fe"]
    ATLASES = ["aal", "schaefer", "yeo7"]   # mix — change freely

    for task_name in TASKS:
        csv_path = f"results/roi_tables/group_{task_name}_multiatlas.csv"
        if not os.path.exists(csv_path):
            print(f"Missing {csv_path}, run extract_roi_multiatlas.py first.")
            continue

        df = pd.read_csv(csv_path)

        for atlas_name in ATLASES:
            print(f"\n[{task_name} | {atlas_name}]")

            atlas_img, label_map = get_atlas(atlas_name)
            tmap = vals2atlas_nilearn(df, atlas_name, atlas_img, label_map)

            out_path = f"results/figures/roi_plots/group_{task_name}_{atlas_name}.png"
            plot_fig(
                tmap, nrow=2, ncol=4, figsize=(16, 8),
                display_mode='z',
                cbar_label=f'{task_name.upper()} QPP — {atlas_name.upper()} atlas',
                save=True, close=True,
                plot_save_path=out_path
            )
            print(f"  Saved -> {out_path}")

    print("\nAll ROI-parcellated brain plots complete!")