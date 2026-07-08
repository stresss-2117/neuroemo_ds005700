"""
Diagnose the black-dot artifacts in MNI-normalized QPP data.
Checks for NaN, Inf, and extreme outlier values.
"""

import nibabel as nib
import numpy as np

sub_id    = "sub-01"
task_name = "fe"
mni_path  = f"results/{task_name}/{sub_id}_qpp_template_MNI.nii.gz"

img  = nib.load(mni_path)
data = img.get_fdata()

print(f"Shape: {data.shape}")
print(f"Data type: {data.dtype}")
print()

n_nan = np.isnan(data).sum()
n_inf = np.isinf(data).sum()
print(f"NaN count : {n_nan}")
print(f"Inf count : {n_inf}")
print()

# Check value distribution
finite_data = data[np.isfinite(data)]
print(f"Min  : {finite_data.min():.4f}")
print(f"Max  : {finite_data.max():.4f}")
print(f"Mean : {finite_data.mean():.4f}")
print(f"Std  : {finite_data.std():.4f}")
print()

# Check percentiles to spot outliers
for p in [50, 90, 95, 99, 99.9, 100]:
    val = np.percentile(np.abs(finite_data), p)
    print(f"  {p}th percentile of |value|: {val:.4f}")

# Check how many voxels are extreme outliers (e.g. >10x the 99th percentile)
p99 = np.percentile(np.abs(finite_data), 99)
extreme = np.abs(data) > (p99 * 5)
print(f"\nVoxels with |value| > 5x(99th percentile): {extreme.sum()}")

# Check zero vs nonzero (background should be exactly 0 after resampling)
n_zero = (data == 0).sum()
n_total = data.size
print(f"\nExact zero voxels: {n_zero} / {n_total} ({100*n_zero/n_total:.1f}%)")