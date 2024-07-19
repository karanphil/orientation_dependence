import argparse
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path

from modules.io import (save_results_as_npz, extract_measures,
                        save_polyfits_as_npz)
from modules.orientation_dependence import (compute_single_fiber_means_new,
                                            fit_single_fiber_results_new,
                                            where_to_patch,
                                            patch_measures)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_peaks',
                   help='Path of the fODF peaks. The peaks are expected to be '
                        'given as unit directions.')
    p.add_argument('in_peak_values',
                   help='Path of the fODF peak values. The peak values must '
                        'not be max-normalized for \neach voxel, but rather '
                        'they should keep the actual fODF amplitude of the '
                        'peaks.')
    p.add_argument('in_fa',
                   help='Path of the FA.')
    p.add_argument('in_nufo',
                   help='Path to the NuFO.')
    p.add_argument('in_wm_mask',
                   help='Path of the WM mask.')
    p.add_argument('out_folder',
                   help='Path of the output folder for txt, png, masks and '
                        'measures.')
    
    p.add_argument('--measures', nargs='+', default=[],
                   action='append', required=True,
                   help='List of measures to characterize.')
    p.add_argument('--measures_names', nargs='+', default=[], action='append',
                   help='List of names for the measures to characterize.')

    p.add_argument('--bundles', nargs='+', default=[],
                   action='append', required=True,
                   help='Path to the bundles masks for where to analyze.')
    p.add_argument('--bundles_names', nargs='+', default=[], action='append',
                   help='List of names for the bundles.')

    g = p.add_argument_group(title='Characterization parameters')
    g.add_argument('--fa_thr', default=0.5,
                   help='Value of FA threshold [%(default)s].')
    g.add_argument('--bin_width_sf', default=1, type=int,
                   help='Value of the bin width for the single-fiber '
                        'characterization [%(default)s].')
    g.add_argument('--bin_width_mf', default=10, type=int,
                   help='Value of the bin width for the multi-fiber '
                        'characterization [%(default)s].')
    g.add_argument('--min_frac_thr', default=0.1,
                   help='Value of the minimal fraction threshold for '
                        'selecting peaks to correct [%(default)s].')
    g.add_argument('--min_nb_voxels', default=1, type=int,
                   help='Value of the minimal number of voxels per bin '
                        '[%(default)s].')
    g.add_argument('--min_corr', default=0.3, type=float,
                   help='Value of the minimal correlation to be eligible for '
                        'patching [%(default)s].')
    
    g1 = p.add_argument_group(title='Polyfit parameters')
    g1.add_argument('--save_polyfit', action='store_true',
                    help='If set, will save the polyfit.')
    g1.add_argument('--use_weighted_polyfit', action='store_true',
                   help='If set, use weights when performing the polyfit. '
                        '[%(default)s].')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    nb_measures = len(args.measures[0])
    if args.measures_names != [] and\
        (len(args.measures_names[0]) != nb_measures):
        parser.error('When using --measures_names, you need to specify ' +
                     'the same number of measures as given in --measures.')

    out_folder = Path(args.out_folder)

    # Load the data
    peaks_img = nib.load(args.in_peaks)
    peak_values_img = nib.load(args.in_peak_values)
    fa_img = nib.load(args.in_fa)
    nufo_img = nib.load(args.in_nufo)
    wm_mask_img = nib.load(args.in_wm_mask)

    peaks = peaks_img.get_fdata()
    peak_values = peak_values_img.get_fdata()
    fa = fa_img.get_fdata()
    nufo = nufo_img.get_fdata()
    wm_mask = wm_mask_img.get_fdata()

    affine = peaks_img.affine

    min_nb_voxels = args.min_nb_voxels

    measures, measures_name = extract_measures(args.measures,
                                               fa.shape,
                                               args.measures_names)

    bundles = []
    bundles_names = []
    for bundle in args.bundles[0]:
        bundles.append(nib.load(bundle).get_fdata())
        bundles_names.append(Path(bundle).name.split(".")[0])

    if args.bundles_names:
        bundles_names = args.bundles_names[0]

    nb_bundles = len(bundles)
    nb_measures = measures.shape[-1]

    nb_bins = int(90 / args.bin_width_sf)
    measure_means = np.zeros((nb_bundles, nb_bins, nb_measures))
    nb_voxels = np.zeros((nb_bundles, nb_bins, nb_measures))
    is_measures = np.ndarray((nb_bundles, nb_bins), dtype=bool)
    pts_origin = np.ndarray((nb_bundles, nb_bins, nb_measures), dtype=object)
    pts_origin.fill("None")

    # For every bundle, compute the mean measures
    for i, (bundle, bundle_name) in enumerate(zip(bundles, bundles_names)):
        print("Computing single-fiber means of bundle {}.".format(bundle_name))
        bins, measure_means[i], nb_voxels[i] =\
            compute_single_fiber_means_new(peaks, fa,
                                           wm_mask,
                                           affine,
                                           measures,
                                           nufo=nufo,
                                           mask=bundle,
                                           bin_width=args.bin_width_sf,
                                           fa_thr=args.fa_thr)

        is_measures[i] = nb_voxels[i, :, 0] >= min_nb_voxels
        pts_origin[i, is_measures[i]] = bundle_name
        nb_filled_bins = np.sum(is_measures[i])
        if nb_filled_bins == 0:
            msg = """No angle bin was filled above the required minimum number
                     of voxels. The script was unable to produce a single-fiber
                     characterization of the measures. If --bundles was used,
                     the region of interest probably contains too few
                     single-fiber voxels. Try to carefully reduce the
                     min_nb_voxels."""
            raise ValueError(msg)

    # For every measure, compute the correlation between bundles
    for i in range(nb_measures):
        print("Computing correlation and patching for measure {}.".format(measures_name[i]))
        to_analyse = measure_means[..., i]
        to_analyse[np.invert(is_measures)] = np.nan
        dataset = pd.DataFrame(data=to_analyse.T)
        corr = dataset.corr()

        for j in range(nb_bundles):
            print("Processing bundle {}".format(bundles_names[j]))
            to_patch = where_to_patch(is_measures[j])
            if np.sum(to_patch) != 0:
                print("Patching bundle {}".format(bundles_names[j]))
                bundle_idx, patchable_pts = patch_measures(to_patch,
                                                           is_measures,
                                                           corr[j],
                                                           min_corr=args.min_corr)
                if bundle_idx == -1:
                    print("WARNING! No bundle found for patching.")
                else:
                    print("Found a bundle for patching: ", bundles_names[bundle_idx])
                    print("Coefficient of correlation is: ", corr[j][bundle_idx])
                    print("Number of points to patch: ", int(np.sum(to_patch)))
                    print("Number of points patched: ", np.sum(patchable_pts))
                    common_pts = is_measures[j] * is_measures[bundle_idx]
                    curr_bundle_mean = np.mean(measure_means[j, ..., i][common_pts])
                    other_bundle_mean = np.mean(measure_means[bundle_idx, ..., i][common_pts])
                    delta_mean = curr_bundle_mean - other_bundle_mean
                    measure_means[j, ..., i][patchable_pts] = measure_means[bundle_idx, ..., i][patchable_pts] + delta_mean
                    nb_voxels[j, ..., i][patchable_pts] = nb_voxels[bundle_idx, ..., i][patchable_pts]
                    pts_origin[j, ..., i][patchable_pts] = bundles_names[bundle_idx]
            out_path = out_folder / (bundles_names[j] + '/1f_results')
            save_results_as_npz(bins, measure_means[j], nb_voxels[j],
                                pts_origin[j], measures_name, out_path)

    if args.use_weighted_polyfit:
        # Why sqrt(n): https://stackoverflow.com/questions/19667877/what-are-the-weight-values-to-use-in-numpy-polyfit-and-what-is-the-error-of-the
        weights = np.sqrt(nb_voxels)
    else:
        weights = None

    new_is_measures = nb_voxels >= min_nb_voxels

    # For every bundle, fit the polynome
    if args.save_polyfit:
        for i, (bundle, bundle_name) in enumerate(zip(bundles, bundles_names)):
            print("Fitting the results of bundle {}.".format(bundle_name))
            measures_fit, measures_max = fit_single_fiber_results_new(bins,
                                                    measure_means[i],
                                                    is_measures=new_is_measures[i],
                                                    weights=weights[i])
            out_path = out_folder / (bundles_names[i] + '/1f_polyfits')
            save_polyfits_as_npz(measures_fit, measures_max, measures_name, out_path)

if __name__ == "__main__":
    main()
