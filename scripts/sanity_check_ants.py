"""
Quick visual check of the ANTs-normalized QPP output.
"""

import nibabel as nib
import matplotlib.pyplot as plt

img = nib.load("results/rest/sub-01_qpp_template_MNI_ANTs.nii.gz")
data = img.get_fdata()

print("Shape:", data.shape)   # should be MNI152 dimensions

mid_frame = data.shape[3] // 2
mid_slice = data.shape[2] // 2

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(data[:, :, mid_slice, mid_frame].T, cmap='RdBu_r', origin='lower')
axes[0].set_title("Axial")

axes[1].imshow(data[:, data.shape[1]//2, :, mid_frame].T, cmap='RdBu_r', origin='lower')
axes[1].set_title("Coronal")

axes[2].imshow(data[data.shape[0]//2, :, :, mid_frame].T, cmap='RdBu_r', origin='lower')
axes[2].set_title("Sagittal")

plt.suptitle("sub-01 rest QPP — ANTs SyN normalized")
plt.tight_layout()
plt.savefig("results/sanity_check_ants.png", dpi=150)
plt.show()
print("Saved -> results/sanity_check_ants.png")