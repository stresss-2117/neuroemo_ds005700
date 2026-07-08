"""
Generate brain plots for QPP results using plot_fig() from the
provided plotting notebook.
"""

import os
import nibabel as nib
from nilearn.image import index_img
from nilearn.datasets import load_mni152_template
from nilearn import plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

template = load_mni152_template(resolution=2)

TASKS = ["rest", "fe"]


def plot_fig(tmap, nrow, ncol, figsize, annotate=False, save=False, close=False,
             display_mode='z', labelsize=12, orientation='vertical',
             cbar_label='QPP BOLD Z-score', plot_save_path='figure.png'):
    """Plot multiple slices of a 3D statistical map. (from your notebook)"""

    if isinstance(tmap, str):
        img = nib.load(tmap)
    else:
        img = tmap

    cut_coords = {
        'x': [int(v) for v in __import__('numpy').linspace(-60, 60, ncol*nrow)],
        'y': [int(v) for v in __import__('numpy').linspace(-100, 70, ncol*nrow)],
        'z': [int(v) for v in __import__('numpy').linspace(-62, 72, ncol*nrow)]
    }

    fig, axes = plt.subplots(nrow, ncol+1, figsize=figsize,
                              gridspec_kw={'width_ratios': [1]*ncol + [0.05]})
    gs = axes[0, ncol].get_gridspec()
    for ax in axes[0:, -1]:
        ax.remove()
    ax_colorbar = fig.add_subplot(gs[0:, -1])

    counter = 0
    for row in range(nrow):
        for col in range(ncol):
            plotting.plot_stat_map(img, title='', black_bg=True, bg_img=template,
                                    axes=axes[row, col], annotate=annotate,
                                    display_mode=display_mode, symmetric_cbar=True,
                                    colorbar=False, cut_coords=[cut_coords[display_mode][counter]])
            counter += 1

    font_properties = {'family': 'serif', 'size': labelsize}
    cmap = cm.get_cmap('cold_hot')
    fdata = img.get_fdata()
    fdata_max = abs(fdata).max()
    vmax_f = fdata_max + (fdata_max * 0.05)
    norm = mpl.colors.Normalize(vmin=-vmax_f, vmax=vmax_f)

    cb1 = mpl.colorbar.ColorbarBase(ax_colorbar, cmap=cmap, norm=norm, orientation=orientation)
    cb1.set_label(cbar_label)
    cb1.ax.tick_params(labelsize=labelsize)

    if save:
        plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    if close:
        plt.close()

    return fig, axes


def plot_subject_qpp(sub_id, task_name, peak_frame=None):
    """Generate a brain plot for one subject's MNI-space QPP."""
    mni_qpp_path = f"results/{task_name}/{sub_id}_qpp_template_MNI.nii.gz"

    if not os.path.exists(mni_qpp_path):
        print(f"  [{sub_id} | {task_name}] No MNI QPP found, skipping.")
        return

    qpp_img = nib.load(mni_qpp_path)
    n_frames = qpp_img.shape[3]

    if peak_frame is None:
        peak_frame = n_frames // 2   # default to middle frame

    tmap_3d = index_img(qpp_img, peak_frame)

    out_path = f"results/{task_name}/{sub_id}_qpp_brainplot.png"
    print(f"  [{sub_id} | {task_name}] Plotting frame {peak_frame}...")

    plot_fig(
        tmap_3d, nrow=2, ncol=4, figsize=(16, 8),
        display_mode='z', cbar_label=f'{sub_id} {task_name} QPP',
        save=True, close=True, plot_save_path=out_path
    )
    print(f"  [{sub_id} | {task_name}] Saved → {out_path}")


if __name__ == "__main__":
    test_subjects = ["sub-01", "sub-23"]
    for sub_id in test_subjects:
        for task_name in TASKS:
            try:
                plot_subject_qpp(sub_id, task_name)
            except Exception as e:
                print(f"  ERROR on {sub_id} | {task_name}: {e}")
                continue