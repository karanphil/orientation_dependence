import numpy as np

from modules.utils import (extend_measure, compute_peaks_fraction)


def analyse_delta_m_max(bins, measure_means_diag, delta_m_max, nb_voxels,
                        frac_thrs=np.array([0.5, 0.6, 0.7, 0.8, 0.9])):

    min_idx = np.nanargmin(measure_means_diag[0], axis=0)
    max_idx = np.nanargmax(measure_means_diag[0], axis=0)

    print(measure_means_diag.shape)

    print(min_idx)
    print(max_idx)

    print(measure_means_diag[0, :, 0])

    print(measure_means_diag[0, :, 0][min_idx[0]])
    print(measure_means_diag[0, :, 0][max_idx[0]])

    return 0

    # min_nb_voxels = nb_voxels[:, min_idx]
    # max_nb_voxels = nb_voxels[:, max_idx]

    mtr_min = mtr_diag[:, min_idx]
    mtr_max = mtr_diag[:, max_idx]
    mtr_delta_m_max = np.zeros(5)
    #mtr_delta_m_max[0] = 0
    mtr_delta_m_max[0:4] = mtr_max - mtr_min
    mtr_delta_m_max[4] = mtr_single_fiber_delta_m_max
    mtr_delta_m_max /= mtr_delta_m_max[4]

    ihmtr_min = ihmtr_diag[:, min_idx]
    ihmtr_max = ihmtr_diag[:, max_idx]
    ihmtr_delta_m_max = np.zeros(5)
    #ihmtr_delta_m_max[0] = 0
    ihmtr_delta_m_max[0:4] = ihmtr_min - ihmtr_max
    ihmtr_delta_m_max[4] = ihmtr_single_fiber_delta_m_max
    ihmtr_delta_m_max /= ihmtr_delta_m_max[4]

    frac_thrs_mid = np.zeros((len(frac_thrs)))
    # frac_thrs_mid[0] = 0
    frac_thrs_mid[-1] = 1
    frac_thrs_mid[0:-1] = (frac_thrs[:-1] + frac_thrs[1:])/2.

    frac_thrs_mid = frac_thrs_mid[~np.isnan(mtr_delta_m_max)]
    mtr_delta_m_max = mtr_delta_m_max[~np.isnan(mtr_delta_m_max)]
    ihmtr_delta_m_max = ihmtr_delta_m_max[~np.isnan(ihmtr_delta_m_max)]

    idx_to_fit = np.array([0, 1, 2, -1])

    mtr_to_fit = np.take(mtr_delta_m_max, idx_to_fit) - 1
    frac_thrs_to_fit = np.take(frac_thrs_mid, idx_to_fit) - 1
    frac_thrs_to_fit = frac_thrs_to_fit[:, np.newaxis]
    slope_mtr, _, _, _ = np.linalg.lstsq(frac_thrs_to_fit, mtr_to_fit)
    origin_mtr = slope_mtr * (-1) + 1

    def mtr_fct(x):
        return slope_mtr * x + origin_mtr
    
    ihmtr_to_fit = np.take(ihmtr_delta_m_max, idx_to_fit) - 1
    slope_ihmtr, _, _, _ = np.linalg.lstsq(frac_thrs_to_fit, ihmtr_to_fit)
    origin_ihmtr = slope_ihmtr * (-1) + 1

    def ihmtr_fct(x):
        return slope_ihmtr * x + origin_ihmtr

    # mtr_fit = np.polyfit(np.take(frac_thrs_mid, idx_to_fit), np.take(mtr_delta_m_max, idx_to_fit), 1)
    # mtr_fit = np.polyfit(frac_thrs_mid[:5], mtr_delta_m_max[:5], 1)
    # mtr_polynome = np.poly1d(mtr_fit)

    # ihmtr_fit = np.polyfit(np.take(frac_thrs_mid, idx_to_fit), np.take(ihmtr_delta_m_max, idx_to_fit), 1)
    # ihmtr_fit = np.polyfit(frac_thrs_mid[:5], ihmtr_delta_m_max[:5], 1)
    # ihmtr_polynome = np.poly1d(ihmtr_fit)
    
    return mtr_fct, ihmtr_fct, mtr_delta_m_max, ihmtr_delta_m_max, frac_thrs_mid

def compute_crossing_fibers_means(peaks, peak_values, wm_mask, affine, nufo,
                                  measures, bin_width=10,
                                  frac_thrs=np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
                                  min_nb_voxels=5):
    peaks_fraction = compute_peaks_fraction(peak_values)

    # Find the direction of the B0 field
    rot = affine[0:3, 0:3]
    z_axis = np.array([0, 0, 1])
    b0_field = np.dot(rot.T, z_axis)

    bins = np.arange(0, 90 + bin_width, bin_width)

    # Calculate the angle between e1 and B0 field
    cos_theta_f1 = np.dot(peaks[..., 0:3], b0_field)
    theta_f1 = np.arccos(cos_theta_f1) * 180 / np.pi
    cos_theta_f2 = np.dot(peaks[..., 3:6], b0_field)
    theta_f2 = np.arccos(cos_theta_f2) * 180 / np.pi

    labels = np.zeros((len(frac_thrs) - 1), dtype=object)
    measure_means = np.zeros((len(frac_thrs) - 1, len(bins) - 1, len(bins) - 1,
                              measures.shape[-1]))
    nb_voxels = np.zeros((len(frac_thrs) - 1, len(bins) - 1, len(bins) - 1))

    for idx in range(len(frac_thrs) - 1):
        # Apply the WM mask
        wm_mask_bool = (wm_mask >= 0.9) & (nufo == 2)
        fraction_mask_bool = (peaks_fraction[..., 0] >= frac_thrs[idx]) & (peaks_fraction[..., 0] < frac_thrs[idx + 1])
        for i in range(len(bins) - 1):
            angle_mask_0_90 = (theta_f1 >= bins[i]) & (theta_f1 < bins[i+1])
            angle_mask_90_180 = (180 - theta_f1 >= bins[i]) & (180 - theta_f1 < bins[i+1])
            angle_mask = angle_mask_0_90 | angle_mask_90_180
            mask_f1 = angle_mask
            for j in range(len(bins) - 1):
                angle_mask_0_90 = (theta_f2 >= bins[j]) & (theta_f2 < bins[j+1]) 
                angle_mask_90_180 = (180 - theta_f2 >= bins[j]) & (180 - theta_f2 < bins[j+1])
                angle_mask = angle_mask_0_90 | angle_mask_90_180
                mask_f2 = angle_mask
                mask = mask_f1 & mask_f2 & wm_mask_bool & fraction_mask_bool
                nb_voxels[idx, i, j] = np.sum(mask)
                if np.sum(mask) < min_nb_voxels:
                    measure_means[idx, i, j, :] = None
                else:
                    measure_means[idx, i, j] = np.mean(measures[mask], axis=0)
        
        for i in range(len(bins) - 1):
            for j in range(i):
                measure_means[idx, i, j] = (measure_means[idx, i, j] + measure_means[idx, j, i]) / 2
                measure_means[idx, j, i] = measure_means[idx, i, j]
                nb_voxels[idx, i, j] = nb_voxels[idx, i, j] + nb_voxels[idx, j, i]
                nb_voxels[idx, j, i] = nb_voxels[idx, i, j]

        labels[idx] = "[" + str(frac_thrs[idx]) + ", " + str(frac_thrs[idx + 1]) + "["

    return bins, measure_means, nb_voxels, labels

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