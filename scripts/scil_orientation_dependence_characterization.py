import argparse
import nibabel as nib
import numpy as np
import logging
import pandas as pd
from pathlib import Path

from scilpy.io.utils import add_verbose_arg

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
    p.add_argument('in_fa',
                   help='Path of the FA.')
    p.add_argument('in_nufo',
                   help='Path to the NuFO.')
    p.add_argument('in_wm_mask',
                   help='Path of the WM mask.')
    p.add_argument('out_folder',
                   help='Path of the output folder for txt, png, masks and '
                        'measures.')
    
    p.add_argument('--measures', nargs='+', default=[], required=True,
                   help='List of measures to characterize.')
    p.add_argument('--measures_names', nargs='+', default=[],
                   help='List of names for the measures to characterize.')

    p.add_argument('--bundles', nargs='+', default=[], required=True,
                   help='Path to the bundles masks for where to analyze.')
    p.add_argument('--bundles_names', nargs='+', default=[],
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

    g1 = p.add_argument_group(title='Patching parameters')
    g1.add_argument('--patch', action='store_true',
                    help='If set, will performing patching of the orientation '
                         'dependence points in order to fill the holes.')
    g1.add_argument('--min_corr', default=0.3, type=float,
                    help='Value of the minimal correlation to be eligible for '
                         'patching [%(default)s].')
    g1.add_argument('--min_frac_pts', default=0.8, type=float,
                    help='Value of the minimal fraction of common points to '
                         'be eligible for patching [%(default)s].')
    
    g2 = p.add_argument_group(title='Polyfit parameters')
    g2.add_argument('--save_polyfit', action='store_true',
                    help='If set, will save the polyfit.')
    g2.add_argument('--use_weighted_polyfit', action='store_true',
                   help='If set, use weights when performing the polyfit. '
                        '[%(default)s].')
    g2.add_argument('--stop_crit', default=0.06, type=float,
                   help='Stopping criteria for the search of polynomial order '
                        '[%(default)s].')

    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    nb_measures = len(args.measures)
    if args.measures_names != [] and\
        (len(args.measures_names) != nb_measures):
        parser.error('When using --measures_names, you need to specify ' +
                     'the same number of measures as given in --measures.')

    out_folder = Path(args.out_folder)

    # Load the data
    peaks_img = nib.load(args.in_peaks)
    fa_img = nib.load(args.in_fa)
    nufo_img = nib.load(args.in_nufo)
    wm_mask_img = nib.load(args.in_wm_mask)

    peaks = peaks_img.get_fdata()
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
    for bundle in args.bundles:
        bundles.append(nib.load(bundle).get_fdata())
        bundles_names.append(Path(bundle).name.split(".")[0])

    if args.bundles_names:
        bundles_names = args.bundles_names

    nb_bundles = len(bundles)
    nb_measures = measures.shape[-1]

    nb_bins = int(90 / args.bin_width_sf)
    measure_means = np.zeros((nb_bundles, nb_bins, nb_measures))
    nb_voxels = np.zeros((nb_bundles, nb_bins, nb_measures))
    is_measures = np.ndarray((nb_bundles, nb_bins), dtype=bool)
    averages = np.zeros((nb_bundles, nb_measures))
    pts_origin = np.ndarray((nb_bundles, nb_bins, nb_measures), dtype=object)
    pts_origin.fill("None")

    # For every bundle, compute the mean measures
    for i, (bundle, bundle_name) in enumerate(zip(bundles, bundles_names)):
        logging.info("Computing single-fiber means of bundle {}.".format(bundle_name))
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
        averages[i] = np.ma.average(np.ma.MaskedArray(measure_means[i],
                                                      mask=np.isnan(measure_means[i])),
                                    weights=nb_voxels[i], axis=0)

    # For every measure, compute the correlation between bundles
    if args.patch:
        for i in range(nb_measures):
            logging.info("Computing correlation and patching for measure {}.".format(measures_name[i]))
            to_analyse = measure_means[..., i]
            to_analyse[np.invert(is_measures)] = np.nan
            dataset = pd.DataFrame(data=to_analyse.T)
            corr = dataset.corr()

            for j in range(nb_bundles):
                logging.info("Processing bundle {}".format(bundles_names[j]))
                to_patch = where_to_patch(is_measures[j])
                if np.sum(to_patch) != 0:
                    logging.info("Patching bundle {}".format(bundles_names[j]))
                    bundle_idx, patchable_pts = patch_measures(to_patch,
                                                            is_measures,
                                                            corr[j],
                                                            min_corr=args.min_corr,
                                                            min_frac_pts=args.min_frac_pts)
                    if bundle_idx == -1:
                        logging.warning("WARNING! No bundle found for patching bundle {}.".format(bundles_names[j]))
                    else:
                        logging.info("Found a bundle for patching: {}".format(bundles_names[bundle_idx]))
                        logging.info("Coefficient of correlation is: {}".format(corr[j][bundle_idx]))
                        logging.info("Number of points to patch: {}".format(int(np.sum(to_patch))))
                        logging.info("Number of points patched: {}".format(np.sum(patchable_pts)))
                        common_pts = is_measures[j] * is_measures[bundle_idx]
                        # Old method, does not take voxel count into account.
                        # curr_bundle_mean = np.mean(measure_means[j, ..., i][common_pts])
                        # other_bundle_mean = np.mean(measure_means[bundle_idx, ..., i][common_pts])
                        curr_bundle_mean = np.average(measure_means[j, ..., i][common_pts],
                                                      weights=nb_voxels[j, ..., i][common_pts])
                        other_bundle_mean = np.average(measure_means[bundle_idx, ..., i][common_pts],
                                                       weights=nb_voxels[bundle_idx, ..., i][common_pts])
                        delta_mean = curr_bundle_mean - other_bundle_mean
                        measure_means[j, ..., i][patchable_pts] = measure_means[bundle_idx, ..., i][patchable_pts] + delta_mean
                        nb_voxels[j, ..., i][patchable_pts] = nb_voxels[bundle_idx, ..., i][patchable_pts]
                        pts_origin[j, ..., i][patchable_pts] = bundles_names[bundle_idx]

    # Saving the results of orientation dependence characterization
    for i in range(nb_bundles):
        out_path = out_folder / (bundles_names[i] + '/1f_results')
        save_results_as_npz(bins, measure_means[i], nb_voxels[i],
                            pts_origin[i], measures_name, out_path)

    new_is_measures = nb_voxels >= min_nb_voxels

    # For every bundle, fit the polynome
    if args.save_polyfit:
        for i, (bundle, bundle_name) in enumerate(zip(bundles, bundles_names)):
            logging.info("Fitting the results of bundle {}.".format(bundle_name))
            averages[i] = np.nanmean(measure_means[i], axis=0)
            measures_fit = fit_single_fiber_results_new(bins,
                                                        measure_means[i],
                                                        is_measures=new_is_measures[i],
                                                        nb_voxels=nb_voxels[i],
                                                        stop_crit=args.stop_crit,
                                                        use_weighted_polyfit=args.use_weighted_polyfit)
            out_path = out_folder / (bundles_names[i] + '/1f_polyfits')
            save_polyfits_as_npz(measures_fit, averages[i], measures_name, out_path)


if __name__ == "__main__":
    main()
