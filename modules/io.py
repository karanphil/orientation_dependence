import nibabel as nib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from modules.utils import compute_peaks_fraction


def save_angle_map(peaks, fa, wm_mask, affine, output_path, fodf_peaks, peak_values,
                    nufo, bin_width=1, fa_thr=0.5):
    # Find the direction of the B0 field
    rot = affine[0:3, 0:3]
    z_axis = np.array([0, 0, 1])
    b0_field = np.dot(rot.T, z_axis)

    # Define the bins
    bins = np.arange(0, 90 + bin_width, bin_width)

    # Calculate the angle between e1 and B0 field
    cos_theta = np.dot(peaks[..., :3], b0_field)
    theta = np.arccos(cos_theta) * 180 / np.pi

    peaks_fraction = compute_peaks_fraction(peak_values)

    cos_theta_f1 = np.dot(fodf_peaks[..., 0:3], b0_field)
    theta_f1 = np.arccos(cos_theta_f1) * 180 / np.pi
    cos_theta_f2 = np.dot(fodf_peaks[..., 3:6], b0_field)
    theta_f2 = np.arccos(cos_theta_f2) * 180 / np.pi
    cos_theta_f3 = np.dot(fodf_peaks[..., 6:9], b0_field)
    theta_f3 = np.arccos(cos_theta_f3) * 180 / np.pi

    peak_1 = np.zeros(wm_mask.shape)
    peak_2 = np.zeros(wm_mask.shape)
    peak_3 = np.zeros(wm_mask.shape)

    # Apply the WM mask and FA threshold
    wm_mask_bool = (wm_mask > 0.9) & (fa > fa_thr) & (nufo == 1)
    for i in range(len(bins) - 1):
        angle_mask_0_90 = (theta >= bins[i]) & (theta < bins[i+1]) 
        angle_mask_90_180 = (180 - theta >= bins[i]) & (180 - theta < bins[i+1])
        angle_mask = angle_mask_0_90 | angle_mask_90_180
        mask = wm_mask_bool & angle_mask
        peak_1[mask] = (bins[i] + bins[i + 1]) /2.
    
    peak_1_sf = np.copy(peak_1)

    wm_mask_bool = (wm_mask > 0.9) & (nufo == 2)
    fraction_mask_bool = (peaks_fraction[..., 0] >= 0.5) & (peaks_fraction[..., 0] < 0.9)
    for i in range(len(bins) - 1):
        angle_mask_0_90 = (theta_f1 >= bins[i]) & (theta_f1 < bins[i+1])
        angle_mask_90_180 = (180 - theta_f1 >= bins[i]) & (180 - theta_f1 < bins[i+1])
        angle_mask = angle_mask_0_90 | angle_mask_90_180
        mask_f1 = angle_mask & fraction_mask_bool & wm_mask_bool
        peak_1[mask_f1] = (bins[i] + bins[i + 1]) /2.

        angle_mask_0_90 = (theta_f2 >= bins[i]) & (theta_f2 < bins[i+1]) 
        angle_mask_90_180 = (180 - theta_f2 >= bins[i]) & (180 - theta_f2 < bins[i+1])
        angle_mask = angle_mask_0_90 | angle_mask_90_180
        mask_f2 = angle_mask & fraction_mask_bool & wm_mask_bool
        peak_2[mask_f2] = (bins[i] + bins[i + 1]) /2.

    wm_mask_bool = (wm_mask > 0.9) & (nufo == 3)
    fraction_mask_bool = (peaks_fraction[..., 0] >= 0.33) & (peaks_fraction[..., 0] < 0.8)
    for i in range(len(bins) - 1):
        angle_mask_0_90 = (theta_f1 >= bins[i]) & (theta_f1 < bins[i+1])
        angle_mask_90_180 = (180 - theta_f1 >= bins[i]) & (180 - theta_f1 < bins[i+1])
        angle_mask = angle_mask_0_90 | angle_mask_90_180
        mask_f1 = angle_mask & wm_mask_bool & fraction_mask_bool
        peak_1[mask_f1] = (bins[i] + bins[i + 1]) /2.

        angle_mask_0_90 = (theta_f2 >= bins[i]) & (theta_f2 < bins[i+1]) 
        angle_mask_90_180 = (180 - theta_f2 >= bins[i]) & (180 - theta_f2 < bins[i+1])
        angle_mask = angle_mask_0_90 | angle_mask_90_180
        mask_f2 = angle_mask & wm_mask_bool & fraction_mask_bool
        peak_2[mask_f2] = (bins[i] + bins[i + 1]) /2.

        angle_mask_0_90 = (theta_f3 >= bins[i]) & (theta_f3 < bins[i+1]) 
        angle_mask_90_180 = (180 - theta_f3 >= bins[i]) & (180 - theta_f3 < bins[i+1])
        angle_mask = angle_mask_0_90 | angle_mask_90_180
        mask_f3 = angle_mask & wm_mask_bool & fraction_mask_bool
        peak_3[mask_f3] = (bins[i] + bins[i + 1]) /2.

    map_1_name = "peak_1_sf_angles_map.nii.gz"
    map_1_path = output_path / "masks" / map_1_name
    nib.save(nib.Nifti1Image(peak_1_sf, affine), map_1_path)

    map_1_name = "peak_1_angles_map.nii.gz"
    map_1_path = output_path / "masks" / map_1_name
    nib.save(nib.Nifti1Image(peak_1, affine), map_1_path)

    map_2_name = "peak_2_angles_map.nii.gz"
    map_2_path = output_path / "masks" / map_2_name
    nib.save(nib.Nifti1Image(peak_2, affine), map_2_path)

    map_3_name = "peak_3_angles_map.nii.gz"
    map_3_path = output_path / "masks" / map_3_name
    nib.save(nib.Nifti1Image(peak_3, affine), map_3_path)