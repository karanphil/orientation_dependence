import numpy as np

from modules.utils import extend_measure

def compute_single_fiber_means(peaks, fa, wm_mask, affine,
                                  measures, nufo=None, mask=None,
                                  bin_width=1, fa_thr=0.5, min_nb_voxels=5):
    # Find the direction of the B0 field
    rot = affine[0:3, 0:3]
    z_axis = np.array([0, 0, 1])
    b0_field = np.dot(rot.T, z_axis)

    bins = np.arange(0, 90 + bin_width, bin_width)

    # Calculate the angle between e1 and B0 field
    cos_theta = np.dot(peaks[..., :3], b0_field)
    theta = np.arccos(cos_theta) * 180 / np.pi

    measure_means = np.zeros((len(bins) - 1, measures.shape[-1]))
    nb_voxels = np.zeros((len(bins) - 1))

    # Apply the WM mask and FA threshold
    if nufo is not None:
        wm_mask_bool = (wm_mask >= 0.9) & (fa > fa_thr) & (nufo == 1)
    else:
        wm_mask_bool = (wm_mask >= 0.9) & (fa > fa_thr)
    if mask is not None:
        wm_mask_bool = wm_mask_bool & (mask > 0)

    for i in range(len(bins) - 1):
        angle_mask_0_90 = (theta >= bins[i]) & (theta < bins[i+1]) 
        angle_mask_90_180 = (180 - theta >= bins[i]) & (180 - theta < bins[i+1])
        angle_mask = angle_mask_0_90 | angle_mask_90_180
        mask_total = wm_mask_bool & angle_mask
        nb_voxels[i] = np.sum(mask_total)
        if np.sum(mask_total) < min_nb_voxels:
            measure_means[i, :] = None
        else:
            measure_means[i] = np.mean(measures[mask_total], axis=0)

    return bins, measure_means, nb_voxels

def fit_single_fiber_results(bins, means, poly_order=8):
    fits = np.ndarray((poly_order + 1, means.shape[-1]))
    for i in range(means.shape[-1]):
        new_bins, new_means = extend_measure(bins, means[..., i])
        mid_bins = (new_bins[:-1] + new_bins[1:]) / 2.
        not_nan = np.isfinite(new_means)
        fits[:, i] = np.polyfit(mid_bins[not_nan], new_means[not_nan],
                                poly_order)
    return fits