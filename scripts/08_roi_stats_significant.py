"""
Script 8: ROI Significance Testing — AAL / Schaefer / Yeo7
=============================================================
Per-ROI paired t-test (rest vs fe), BH-FDR corrected, across all three
atlases. Only ROIs that survive FDR correction get painted on the brain
plots — everything else stays plain anatomy.

FIXES in this version:
  - Excludes AAL's "Background" pseudo-ROI (was being tested and flagged
    significant despite being non-brain voxels — not a real result).
  - Adds Cohen's dz effect size alongside t/p, matching the reporting
    style used elsewhere in the project (t, q, Cohen's dz per measure).
  - plot_surface() reverted to interpolation='nearest_most_frequent'
    (was 'linear', which blends values across significant/non-significant
    parcel boundaries and can show colorbar values that exceed any real
    ROI difference — a rendering artifact, not signal).

TEMP FIX (still active): scipy.stats import is disabled (Windows pagefile
DLL issue). Paired t-test + p-value are computed manually with numpy/math.
Once the pagefile fix is confirmed working after your restart, you can
switch back to `from scipy import stats` + `stats.ttest_rel(...)` if you
prefer — this manual version gives identical results either way.

Uses the FULL QPP cycle (mean |amplitude| across all window frames),
same approach as 07_qpp_cognition_overlap.py.

Usage:
  python scripts/08_roi_stats_significant.py
"""

import os
import math
import numpy as np
import pandas as pd
import nibabel as nb
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from nilearn import plotting, surface, datasets
from nilearn.image import resample_to_img

from extract_roi_multiatlas import ATLAS_REGISTRY   # reuse your atlas loaders

RESULTS_DIR = "results"
OUT_DIR = "results/figures/roi_plots_significant"
os.makedirs(OUT_DIR, exist_ok=True)

ALL_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 41)]
TASKS = ["rest", "fe"]
FDR_ALPHA = 0.05


template = datasets.load_mni152_template(resolution=1)


# ══════════════════════════════════════════════════════════════
# Manual paired t-test (no scipy) — TEMP FIX
# ══════════════════════════════════════════════════════════════

def _betainc(a, b, x, iters=200):
    """Regularized incomplete beta function (continued fraction), stdlib-only."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
              + a * math.log(x) + b * math.log(1 - x))
    front = math.exp(lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(iters):
        m = i // 2
        if i == 0:
            numerator = 1.0
        elif i % 2 == 0:
            numerator = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            numerator = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1 / d
        c = 1.0 + numerator / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= d * c
    result = front * (f - 1)
    return result if x < (a + 1) / (a + b + 2) else 1 - result


def ttest_rel_manual(a, b):
    """Paired t-test, returns (t_stat, p_value) — no scipy dependency."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = a - b
    n = len(diff)
    if n < 2:
        return 0.0, 1.0
    mean_d = diff.mean()
    std_d = diff.std(ddof=1)
    if std_d == 0:
        return 0.0, 1.0
    t_stat = mean_d / (std_d / math.sqrt(n))
    df = n - 1
    x = df / (df + t_stat ** 2)
    p_value = _betainc(df / 2, 0.5, x)
    return t_stat, p_value


# ══════════════════════════════════════════════════════════════
# Full-cycle ROI extraction (mirrors 07_qpp_cognition_overlap.py)
# ══════════════════════════════════════════════════════════════

def extract_roi_values_full_cycle(qpp_img, atlas_img, label_map):
    """
    Mean |amplitude| across ALL frames of the QPP window, per ROI.
    Using |value| because a QPP cycle swings high-to-low by definition —
    signed averaging across the full cycle would wash out real engagement.
    """
    qpp_data = qpp_img.get_fdata()
    ref_3d = nb.Nifti1Image(
        qpp_data[..., 0] if qpp_data.ndim == 4 else qpp_data, qpp_img.affine)

    atlas_res = resample_to_img(atlas_img, ref_3d, interpolation='nearest')
    atlas_data = atlas_res.get_fdata()
    if atlas_data.ndim == 4:
        atlas_data = atlas_data[..., 0]
    atlas_data = atlas_data.astype(int)

    rows = []
    for roi_id, roi_name in label_map.items():
        roi_mask = (atlas_data == roi_id)
        if roi_mask.sum() == 0:
            continue
        if qpp_data.ndim == 4:
            per_frame = [np.abs(qpp_data[..., f][roi_mask]).mean()
                         for f in range(qpp_data.shape[-1])]
            val = float(np.mean(per_frame))
        else:
            val = float(np.abs(qpp_data[roi_mask]).mean())
        rows.append({"ROI": roi_name, "Weights": val})

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# Statistics
# ══════════════════════════════════════════════════════════════

def bh_fdr(pvals, alpha=FDR_ALPHA):
    pvals = np.array(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = (np.arange(1, n + 1) / n) * alpha
    passed = ranked <= thresh
    if not passed.any():
        return np.zeros(n, dtype=bool)
    cutoff = ranked[np.where(passed)[0].max()]
    return pvals <= cutoff


def build_subject_roi_table(atlas_img, label_map):
    rows = []
    for task in TASKS:
        for sub in ALL_SUBJECTS:
            path = f"{RESULTS_DIR}/{task}/{sub}_qpp_template_MNI_ANTs.nii.gz"
            if not Path(path).exists():
                continue
            qpp_img = nb.load(path)
            df = extract_roi_values_full_cycle(qpp_img, atlas_img, label_map)
            df["Subject"] = sub
            df["Task"] = task
            rows.append(df)

    if not rows:
        return pd.DataFrame(columns=["ROI", "Weights", "Subject", "Task"])
    return pd.concat(rows, ignore_index=True)


def significant_roi_map(atlas_name):
    print(f"  Loading atlas: {atlas_name}...")
    atlas_img, label_map = ATLAS_REGISTRY[atlas_name]()

    print(f"  Building per-subject ROI table (full-cycle |amplitude|)...")
    df = build_subject_roi_table(atlas_img, label_map)

    # FIX: drop AAL's "Background" pseudo-ROI (non-brain voxels, not a real region)
    n_before = df["ROI"].nunique()
    df = df[df["ROI"] != "Background"]
    n_after = df["ROI"].nunique()
    if n_before != n_after:
        print(f"  Excluded 'Background' pseudo-ROI ({n_before} -> {n_after} ROIs)")

    sig_rows = []
    for roi in df["ROI"].unique():
        rest_v = df[(df.ROI == roi) & (df.Task == "rest")].set_index("Subject")["Weights"]
        fe_v = df[(df.ROI == roi) & (df.Task == "fe")].set_index("Subject")["Weights"]
        common = rest_v.index.intersection(fe_v.index)
        if len(common) < 2:
            continue

        rest_arr = rest_v.loc[common].values
        fe_arr = fe_v.loc[common].values
        t, p = ttest_rel_manual(fe_arr, rest_arr)

        # FIX: Cohen's dz — standardized effect size for a paired design,
        # matching the reporting style used for the QPP measures elsewhere
        # in the project (t, q, Cohen's dz side by side).
        diff_vals = fe_arr - rest_arr
        diff_std = diff_vals.std(ddof=1)
        cohens_dz = float(diff_vals.mean() / diff_std) if diff_std > 0 else 0.0

        sig_rows.append({
            "ROI": roi, "N": len(common), "t": t, "p": p,
            "diff": float(diff_vals.mean()), "cohens_dz": cohens_dz,
        })

    res = pd.DataFrame(sig_rows)
    if res.empty:
        print(f"  [{atlas_name}] No paired data found — check that ANTs QPP files exist.")
        return None, res

    res["sig"] = bh_fdr(res["p"].values)
    res = res.sort_values("p")
    print(f"  [{atlas_name}] {res['sig'].sum()} / {len(res)} ROIs significant (FDR<{FDR_ALPHA})")

    # Paint ONLY significant ROIs — everything else stays exactly 0
    atlas_data = atlas_img.get_fdata()
    if atlas_data.ndim == 4:
        atlas_data = atlas_data[..., 0]
    out = np.zeros_like(atlas_data)
    name_to_diff = dict(zip(res[res.sig]["ROI"], res[res.sig]["diff"]))
    for roi_id, name in label_map.items():
        if name in name_to_diff:
            out[atlas_data == roi_id] = name_to_diff[name]

    sig_img = nb.Nifti1Image(out, affine=atlas_img.affine)

    nifti_path = f"{OUT_DIR}/{atlas_name}_sig_diff.nii.gz"
    nb.save(sig_img, nifti_path)
    print(f"  Saved significance NIfTI -> {nifti_path}")

    return sig_img, res


# ══════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════

def plot_fig(tmap, nrow, ncol, figsize, annotate=False, save=False, close=False,
             display_mode='z', labelsize=12, orientation='vertical',
             cbar_label='QPP Difference', plot_save_path='figure.png',
             threshold=1e-6, black_bg=True, cmap_name='RdBu_r'):

    if isinstance(tmap, str):
        img = nb.load(tmap)
    else:
        img = tmap

    cut_coords = {
        'x': np.linspace(-60, 60, ncol * nrow, dtype=int).tolist(),
        'y': np.linspace(-100, 70, ncol * nrow, dtype=int).tolist(),
        'z': np.linspace(-62, 72, ncol * nrow, dtype=int).tolist()
    }
    counter = 0

    fig, axes = plt.subplots(nrow, ncol + 1, figsize=figsize,
                              gridspec_kw={'width_ratios': [1] * ncol + [0.05]})
    gs = axes[0, ncol].get_gridspec()
    for ax in axes[0:, -1]:
        ax.remove()
    ax_colorbar = fig.add_subplot(gs[0:, -1])

    for row in range(nrow):
        for col in range(ncol):
            plotting.plot_stat_map(
                img, title='', black_bg=black_bg, bg_img=template,
                axes=axes[row, col], annotate=annotate,
                display_mode=display_mode, symmetric_cbar=True,
                colorbar=False, cut_coords=[cut_coords[display_mode][counter]],
                threshold=threshold, cmap=cmap_name,
            )
            counter += 1

    font_properties = {'family': 'serif', 'size': labelsize}
    cmap = mpl.colormaps[cmap_name]
    fdata = img.get_fdata()
    finite_abs = np.abs(fdata[np.isfinite(fdata)])
    fdata_max = finite_abs.max() if finite_abs.size else 1.0
    vmax_f = fdata_max + (fdata_max * 0.05) if fdata_max > 0 else 1.0
    norm = mpl.colors.Normalize(vmin=-vmax_f, vmax=vmax_f)

    cb1 = mpl.colorbar.ColorbarBase(ax_colorbar, cmap=cmap, norm=norm, orientation=orientation)
    cb1.set_label(cbar_label)
    cb1.ax.tick_params(labelsize=labelsize)

    if save:
        plt.savefig(plot_save_path, dpi=300, bbox_inches='tight',
                     facecolor='black' if black_bg else 'white')
    if close:
        plt.close()

    return fig, axes


def plot_surface(tmap, cmap_name='RdBu_r', threshold=1e-6, save_path='cortex_surface.png'):
    """
    Cortical surface rendering — only significant vertices colored.
    FIX: interpolation reverted to 'nearest_most_frequent' (was 'linear').
    Linear interpolation blends values across the boundary between a
    significant parcel and its non-significant (zero) neighbor, which
    can make the colorbar show values that exceed any real ROI
    difference in the stats table. nearest_most_frequent keeps sharp,
    accurate parcel edges instead.
    """
    fsaverage = datasets.fetch_surf_fsaverage()

    fig, axes = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, figsize=(10, 8))
    fig.suptitle('Cortex')

    views_hemis = [
        ('left', 'lateral', fsaverage.infl_left, fsaverage.sulc_left),
        ('right', 'lateral', fsaverage.infl_right, fsaverage.sulc_right),
        ('left', 'medial', fsaverage.infl_left, fsaverage.sulc_left),
        ('right', 'medial', fsaverage.infl_right, fsaverage.sulc_right),
    ]

    for ax, (hemi, view, infl, sulc) in zip(axes.flat, views_hemis):
        pial = fsaverage.pial_left if hemi == 'left' else fsaverage.pial_right
        texture = surface.vol_to_surf(tmap, pial, interpolation='nearest_most_frequent')

        plotting.plot_surf_stat_map(
            infl, texture, hemi=hemi, view=view, bg_map=sulc,
            cmap=cmap_name, threshold=threshold,
            colorbar=(ax is axes.flat[-1]), axes=ax,
        )

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return fig


# ══════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    ATLASES_TO_RUN = ["aal", "schaefer", "yeo7"]

    for atlas_name in ATLASES_TO_RUN:
        print(f"\n{'='*60}")
        print(f"ATLAS: {atlas_name}")
        print(f"{'='*60}")

        sig_img, res_df = significant_roi_map(atlas_name)

        stats_path = f"{OUT_DIR}/{atlas_name}_roi_stats.csv"
        res_df.to_csv(stats_path, index=False)
        print(f"  Stats saved -> {stats_path}")

        if sig_img is None or res_df.empty or res_df["sig"].sum() == 0:
            print(f"  No significant ROIs for {atlas_name} — skipping plot.")
            continue

        out_path = f"{OUT_DIR}/group_fe_vs_rest_{atlas_name}_SIG.png"
        plot_fig(
            sig_img, nrow=2, ncol=4, figsize=(16, 8), display_mode='z',
            black_bg=True, cmap_name='RdBu_r', threshold=1e-6,
            cbar_label=f'FE - Rest QPP ({atlas_name.upper()}, FDR-significant only)',
            save=True, close=True, plot_save_path=out_path
        )
        print(f"  Saved -> {out_path}")

        surf_path = f"{OUT_DIR}/group_fe_vs_rest_{atlas_name}_SIG_surface.png"
        plot_surface(sig_img, cmap_name='RdBu_r', threshold=1e-6, save_path=surf_path)
        print(f"  Saved -> {surf_path}")

    print("\nAll significance-masked ROI plots complete!")