import numpy as np


def compute_corrections(polynome, angle, fraction, fraction_factor):
    bins = np.arange(0, 90 + 1, 1)
    max_poly = np.max(polynome(bins))
    correction = fraction * (max_poly - polynome(angle)) * fraction_factor
    return correction

def compute_peaks_fraction(peak_values):
    peak_values_sum = np.sum(peak_values, axis=-1)
    peak_values_sum = np.repeat(peak_values_sum.reshape(peak_values_sum.shape + (1,)),
                                peak_values.shape[-1], axis=-1)
    peaks_fraction = peak_values / peak_values_sum
    return peaks_fraction

def extend_measure(bins, measure):
    new_bins = np.concatenate((np.flip(-bins[1:10]), bins, 180 - np.flip(bins[-10:-1])))
    new_measure = np.concatenate((np.flip(measure[1:10]), measure, np.flip(measure[-10:-1])))
    return new_bins, new_measure

def nb_peaks_factor(polynome, peak_fraction):
    nb_peaks_factor = polynome(peak_fraction)
    return np.clip(nb_peaks_factor, 0, 1)