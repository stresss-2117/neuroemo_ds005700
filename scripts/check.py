import nibabel as nib
import matplotlib.pyplot as plt

img = nib.load("results/rest/sub-01_qpp_template_MNI.nii.gz")
data = img.get_fdata()
print("Shape:", data.shape)   # should be (X, Y, Z, 10) in MNI dimensions

mid_frame = data.shape[3] // 2
mid_slice = data.shape[2] // 2

plt.imshow(data[:, :, mid_slice, mid_frame].T, cmap='RdBu_r', origin='lower')
plt.title("sub-01 rest QPP — MNI space, middle slice/frame")
plt.colorbar()
plt.savefig("results/sanity_check_mni.png", dpi=150)
plt.show()
print("Saved sanity check → results/sanity_check_mni.png")