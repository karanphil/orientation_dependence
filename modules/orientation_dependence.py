import numpy as np

from scipy.stats import (shapiro, kstest)

from modules.utils import (extend_measure, compute_peaks_fraction,
                           compute_corrections, nb_peaks_factor,
                           extend_measure_v2, extend_measure_v3)


def analyse_delta_m_max(bins, means_diag, sf_delta_m_max, nb_voxels,
                        frac_thrs=np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
                        min_nb_voxels_to_fit=400):

    min_idx = np.nanargmin(means_diag[0], axis=0)
    max_idx = np.nanargmax(means_diag[0], axis=0)
    measures_idx = np.arange(means_diag.shape[-1])
    means_min = means_diag[:, min_idx, measures_idx]
    means_max = means_diag[:, max_idx, measures_idx]
    min_bins = (bins[min_idx] + bins[min_idx + 1]) /2 # +1 might cause problems
    max_bins = (bins[max_idx] + bins[max_idx + 1]) /2 # for last index

    delta_m_max = np.zeros((len(frac_thrs), means_diag.shape[-1]))
    delta_m_max[0:4] = means_max - means_min
    delta_m_max[4] = sf_delta_m_max
    delta_m_max /= delta_m_max[4]

    frac_thrs_mid = np.zeros((len(frac_thrs)))
    frac_thrs_mid[-1] = 1
    frac_thrs_mid[0:-1] = (frac_thrs[:-1] + frac_thrs[1:])/2.

    frac_idx = np.arange(len(frac_thrs) - 1)

    slope = np.zeros((means_diag.shape[-1]))
    origin = np.zeros((means_diag.shape[-1]))
    for i in range(means_diag.shape[-1]):
        min_nb_voxels = nb_voxels[:, min_idx[i]]
        max_nb_voxels = nb_voxels[:, max_idx[i]]
        nb_voxels_check = (min_nb_voxels >= min_nb_voxels_to_fit) & (max_nb_voxels >= min_nb_voxels_to_fit)
        idx_to_fit = frac_idx[nb_voxels_check]
        idx_to_fit =  np.concatenate((idx_to_fit, [-1]))
        delta_m_max_to_fit = np.take(delta_m_max[:, i], idx_to_fit) - 1
        frac_thrs_to_fit = np.take(frac_thrs_mid, idx_to_fit) - 1
        frac_thrs_to_fit = frac_thrs_to_fit[~np.isnan(delta_m_max_to_fit)]
        delta_m_max_to_fit = delta_m_max_to_fit[~np.isnan(delta_m_max_to_fit)]
        frac_thrs_to_fit = frac_thrs_to_fit[:, np.newaxis]
        slope[i], _, _, _ = np.linalg.lstsq(frac_thrs_to_fit, 
                                            delta_m_max_to_fit, rcond=None)
        origin[i] = slope[i] * (-1) + 1

    return slope, origin, delta_m_max, frac_thrs_mid, min_bins, max_bins


def correct_measure(peaks, peak_values, measure, affine, wm_mask,
                    polynome, peak_frac_thr=0, delta_m_max_fct=None,
                    mask=None):
    peaks_fraction = compute_peaks_fraction(peak_values)

    if delta_m_max_fct is not None:
        peaks_fraction_factor = nb_peaks_factor(delta_m_max_fct, peaks_fraction[..., 0])
    else:
        peaks_fraction_factor = np.ones(peaks_fraction.shape[:3])
    
    # Find the direction of the B0 field
    rot = affine[0:3, 0:3]
    z_axis = np.array([0, 0, 1])
    b0_field = np.dot(rot.T, z_axis)

    peaks_angles = np.empty((peaks_fraction.shape))
    peaks_angles[:] = np.nan
    corrections = np.zeros((peaks_fraction.shape))
    # Calculate the angle between e1 and B0 field for each peak
    wm_mask_bool = (wm_mask >= 0.9)
    if mask is not None:
        wm_mask_bool = wm_mask_bool & (mask > 0)
    for i in range(peaks_angles.shape[-1]):
        mask = wm_mask_bool & (peaks_fraction[..., i] > peak_frac_thr)
        cos_theta = np.dot(peaks[mask, i*3:(i+1)*3], b0_field)
        theta = np.arccos(cos_theta) * 180 / np.pi
        peaks_angles[mask, i] = np.abs(theta//90 * 90 - theta%90) % 180

        corrections[mask, i] = compute_corrections(polynome,
                                                   peaks_angles[mask, i],
                                                   peaks_fraction[mask, i],
                                                   peaks_fraction_factor[mask])
    
    total_corrections = np.sum(corrections, axis=-1)

    return measure + total_corrections


def compute_three_fibers_means(peaks, peak_values, wm_mask, affine, nufo,
                               measures, bin_width=30, mask=None,
                               frac_thrs=np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
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
    cos_theta_f3 = np.dot(peaks[..., 6:9], b0_field)
    theta_f3 = np.arccos(cos_theta_f3) * 180 / np.pi

    labels = np.zeros((len(frac_thrs) - 1), dtype=object)
    measure_means = np.zeros((len(frac_thrs) - 1, len(bins) - 1,
                          measures.shape[-1]))
    nb_voxels = np.zeros((len(frac_thrs) - 1, len(bins) - 1))

    for idx in range(len(frac_thrs) - 1):
        # Apply the WM mask
        wm_mask_bool = (wm_mask >= 0.9) & (nufo == 3)
        if mask is not None:
            wm_mask_bool = wm_mask_bool & (mask > 0)
        fraction_mask_bool = (peaks_fraction[..., 0] >= frac_thrs[idx]) & (peaks_fraction[..., 0] < frac_thrs[idx + 1])
        for i in range(len(bins) - 1):
            angle_mask_0_90 = (theta_f1 >= bins[i]) & (theta_f1 < bins[i+1])
            angle_mask_90_180 = (180 - theta_f1 >= bins[i]) & (180 - theta_f1 < bins[i+1])
            angle_mask = angle_mask_0_90 | angle_mask_90_180
            mask_f1 = angle_mask

            angle_mask_0_90 = (theta_f2 >= bins[i]) & (theta_f2 < bins[i+1]) 
            angle_mask_90_180 = (180 - theta_f2 >= bins[i]) & (180 - theta_f2 < bins[i+1])
            angle_mask = angle_mask_0_90 | angle_mask_90_180
            mask_f2 = angle_mask

            angle_mask_0_90 = (theta_f3 >= bins[i]) & (theta_f3 < bins[i+1]) 
            angle_mask_90_180 = (180 - theta_f3 >= bins[i]) & (180 - theta_f3 < bins[i+1])
            angle_mask = angle_mask_0_90 | angle_mask_90_180
            mask_f3 = angle_mask

            mask = mask_f1 & mask_f2 & mask_f3 & wm_mask_bool & fraction_mask_bool
            nb_voxels[idx, i] = np.sum(mask)
            if np.sum(mask) < min_nb_voxels:
                measure_means[idx, i, :] = None
            else:
                measure_means[idx, i] = np.mean(measures[mask], axis=0)

        labels[idx] = "[" + str(frac_thrs[idx]) + ", " + str(frac_thrs[idx + 1]) + "["

    return bins, measure_means, nb_voxels, labels


def compute_two_fibers_means(peaks, peak_values, wm_mask, affine, nufo,
                             measures, bin_width=10, mask=None,
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
        if mask is not None:
            wm_mask_bool = wm_mask_bool & (mask > 0)
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
                               bin_width=1, fa_thr=0.5):
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
        if np.sum(mask_total) < 1:
            measure_means[i, :] = None
        else:
            measure_means[i] = np.mean(measures[mask_total], axis=0)

    return bins, measure_means, nb_voxels


def fit_single_fiber_results(bins, means, poly_order=10, is_measures=None,
                             weights=None):
    if is_measures is None:
        is_measures = np.ones(means.shape[0])
    if weights is None:
        weights = np.ones(means.shape[0])
    fits = np.zeros((poly_order + 1, means.shape[-1]))
    residuals = np.zeros((means.shape[-1]), dtype=object)
    for i in range(means.shape[-1]):
        new_bins, new_means, new_is_measures, new_weights =\
            extend_measure_v3(bins, means[..., i], is_measure=is_measures,
                              weights=weights)
        # mid_bins = (new_bins[:-1] + new_bins[1:]) / 2.
        effective_poly_order = int(np.floor(poly_order * (new_bins[-2] - new_bins[1]) / (bins[-1] - bins[1])))
        print("Polyfit order was set to", effective_poly_order)
        fits[poly_order - effective_poly_order:, i], residuals[i], _, _, _ =\
            np.polyfit(new_bins[new_is_measures],
                       new_means[new_is_measures],
                       effective_poly_order,
                       w=new_weights[new_is_measures],
                       full=True)
    return fits, residuals