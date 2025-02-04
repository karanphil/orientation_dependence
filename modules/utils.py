import numpy as np


def compute_corrections(polynome, angle, fraction):
    bins = np.arange(0, 90 + 1, 1)
    max_poly = np.max(polynome(bins))
    correction = fraction * (max_poly - polynome(angle))
    return correction


def compute_peaks_fraction(peak_values):
    peak_values_sum = np.sum(peak_values, axis=-1)
    peak_values_sum = np.repeat(peak_values_sum.reshape(peak_values_sum.shape + (1,)),
                                peak_values.shape[-1], axis=-1)
    peaks_fraction = peak_values / peak_values_sum
    return peaks_fraction


def extend_measure_v3(bins, measure, is_measure=None, weights=None):
    # new_bins = np.concatenate((np.flip(-bins[1:10]), bins, 180 - np.flip(bins[-10:-1])))
    # new_measure = np.concatenate((np.flip(measure[1:10]), measure, np.flip(measure[-10:-1])))
    new_bins = np.concatenate((np.flip(-bins[1:]), bins, 180 - np.flip(bins[:-1])))
    new_measure = np.concatenate((np.flip(measure[1:]), measure, np.flip(measure[:-1])))
    new_is_measure = np.ones(new_measure.shape[0])
    new_weights = np.ones(new_measure.shape[0])
    if is_measure is not None:
        new_is_measure = np.concatenate((np.flip(is_measure[1:]),
                                         is_measure, np.flip(is_measure[:-1])))
    if weights is not None:
        new_weights = np.concatenate((np.flip(weights[1:]),
                                      weights, np.flip(weights[:-1])))
    return new_bins[1: -1], new_measure, new_is_measure, new_weights


def extend_measure_v2(bins, measure, is_measure=None, weights=None):
    new_bins = bins
    new_measure = measure
    new_is_measure = is_measure
    new_weights = weights
    return new_bins, new_measure, new_is_measure, new_weights


def extend_measure(bins, measure, is_measure, weights=None):
    mid_bins = (bins[:-1] + bins[1:]) / 2.
    new_bins = mid_bins[is_measure][~np.isnan(measure[is_measure])]
    bin_width = bins[1] - bins[0]
    new_measure = measure[is_measure][~np.isnan(measure[is_measure])]
    new_is_measure = is_measure[is_measure][~np.isnan(measure[is_measure])]
    new_weights = weights[is_measure][~np.isnan(measure[is_measure])]
    new_bins = np.concatenate(([new_bins[0] - bin_width], new_bins,
                               [new_bins[-1] + bin_width]))
    new_measure = np.concatenate(([new_measure[0]], new_measure,
                                  [new_measure[-1]]))
    new_is_measure = np.concatenate(([new_is_measure[0]], new_is_measure,
                                     [new_is_measure[-1]]))
    new_weights = np.concatenate(([new_weights[0]], new_weights,
                                  [new_weights[-1]]))
    return new_bins, new_measure, new_is_measure, new_weights


def nb_peaks_factor(delta_m_max_fct, peak_fraction):
    nb_peaks_factor = delta_m_max_fct(peak_fraction)
    return np.clip(nb_peaks_factor, 0, 1)


def compute_sf_mf_mask(fixel_density_masks):
    fixel_mask = np.clip(np.sum(fixel_density_masks, axis=-1), 0, 1)
    sf_mask = np.where(np.sum(fixel_mask, axis=-1) == 1, 1, 0)
    mf_mask = np.where(np.sum(fixel_mask, axis=-1) > 1, 1, 0)
    return sf_mask, mf_mask
