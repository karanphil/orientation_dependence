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
    p.add_argument('in_fixel_density_masks',
                   help='Path of the fixel density masks. This is the output '
                        'of the scil_bundle_fixel_analysis script, without '
                        'the split_bundles option. Thus, all the bundles '
                        'be present in the file, as a 5th dimension.')
    p.add_argument('out_folder',
                   help='Path of the output folder for txt, png, masks and '
                        'measures.')
    
    p.add_argument('--measures', nargs='+', default=[], required=True,
                   help='List of measures to characterize.')
    p.add_argument('--measures_names', nargs='+', default=[],
                   help='List of names for the measures to characterize.')

    p.add_argument('--bundles_names', nargs='+', default=[],
                   help='List of names for the bundles.')

    p.add_argument('--lookuptable',
                   help='Path of the bundles lookup table, outputed by the '
                        'scil_fixel_density_maps script. Allows to make sure '
                        'the polyfits and fixel_density_maps follow the same '
                        'order.')    

    g = p.add_argument_group(title='Characterization parameters')
    g.add_argument('--bin_width_mf', default=10, type=int,
                   help='Value of the bin width for the multi-fiber '
                        'characterization [%(default)s].')
    g.add_argument('--min_frac_thr', default=0.1,
                   help='Value of the minimal fraction threshold for '
                        'selecting peaks to correct [%(default)s].')
    g.add_argument('--min_nb_voxels', default=1, type=int,
                   help='Value of the minimal number of voxels per bin '
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
    peaks = peaks_img.get_fdata()
    affine = peaks_img.affine

    fixel_density_masks_img = nib.load(args.in_fixel_density_masks)
    fixel_density_masks = fixel_density_masks_img.get_fdata()

    if args.lookuptable:
        lookuptable = np.loadtxt(args.lookuptable, dtype=str)[0]

    min_nb_voxels = args.min_nb_voxels

    measures, measures_name = extract_measures(args.measures,
                                               peaks.shape[0:3],
                                               args.measures_names)

    if args.bundles_names:
        bundles_names = args.bundles_names

    nb_bundles = len(bundles_names)
    nb_measures = measures.shape[-1]

    nb_bins = int(90 / args.bin_width_mf)
    measure_means = np.zeros((nb_bundles, nb_bins, nb_measures))
    measure_stds = np.zeros((nb_bundles, nb_bins, nb_measures))
    nb_voxels = np.zeros((nb_bundles, nb_bins, nb_measures))
    is_measures = np.ndarray((nb_bundles, nb_bins), dtype=bool)

    # For every bundle, compute the mean measures
    for i, bundle_name in enumerate(bundles_names):
        logging.info("Computing multi-fiber means of bundle {}.".format(bundle_name))
        if args.lookuptable:
                if bundle_name in lookuptable:
                    bundle_idx = np.argwhere(lookuptable == bundle_name)[0][0]
                else:
                    raise ValueError("Polyfit from bundle not present in lookup table.")
        else:
            bundle_idx = i
        mf_bunlde_mask = np.where(np.sum(fixel_density_masks[..., bundle_idx], axis=-1) >= 1, 1, 0)
        # print(fixel_density_masks[..., bundle_idx])
        bundle_peaks = peaks[fixel_density_masks[..., bundle_idx]]
    #     bins, measure_means[i], measure_stds[i], nb_voxels[i] =\
    #         compute_single_fiber_means_new(peaks, fa,
    #                                        wm_mask,
    #                                        affine,
    #                                        measures,
    #                                        nufo=nufo,
    #                                        mask=bundle,
    #                                        bin_width=args.bin_width_sf,
    #                                        fa_thr=args.fa_thr)

    #     is_measures[i] = nb_voxels[i, :, 0] >= min_nb_voxels
    #     pts_origin[i, is_measures[i]] = bundle_name
    #     nb_filled_bins = np.sum(is_measures[i])
    #     if nb_filled_bins == 0:
    #         msg = """No angle bin was filled above the required minimum number
    #                  of voxels. The script was unable to produce a single-fiber
    #                  characterization of the measures. If --bundles was used,
    #                  the region of interest probably contains too few
    #                  single-fiber voxels. Try to carefully reduce the
    #                  min_nb_voxels."""
    #         raise ValueError(msg)
    #     averages[i] = np.ma.average(np.ma.MaskedArray(measure_means[i],
    #                                                   mask=np.isnan(measure_means[i])),
    #                                 weights=nb_voxels[i], axis=0)

    # # Saving the results of orientation dependence characterization
    # for i in range(nb_bundles):
    #     out_path = out_folder / (bundles_names[i] + '/1f_results')
    #     save_results_as_npz(bins, measure_means[i], measure_stds[i], nb_voxels[i],
    #                         pts_origin[i], measures_name, out_path)

    # new_is_measures = nb_voxels >= min_nb_voxels

    # # For every bundle, fit the polynome
    # if args.save_polyfit:
    #     for i, (bundle, bundle_name) in enumerate(zip(bundles, bundles_names)):
    #         logging.info("Fitting the results of bundle {}.".format(bundle_name))
    #         # Taking the average of the mean points instead of all the points
    #         # averages[i] = np.nanmean(measure_means[i], axis=0)
    #         measures_fit, measures_max = fit_single_fiber_results_new(bins,
    #                                                     measure_means[i],
    #                                                     is_measures=new_is_measures[i],
    #                                                     nb_voxels=nb_voxels[i],
    #                                                     stop_crit=args.stop_crit,
    #                                                     use_weighted_polyfit=args.use_weighted_polyfit)
    #         out_path = out_folder / (bundles_names[i] + '/1f_polyfits')
    #         if args.reference == "mean":
    #             save_polyfits_as_npz(measures_fit, averages[i],
    #                                  measures_name, out_path)
    #         elif args.reference == "maximum":
    #             save_polyfits_as_npz(measures_fit, measures_max,
    #                                  measures_name, out_path)


if __name__ == "__main__":
    main()
