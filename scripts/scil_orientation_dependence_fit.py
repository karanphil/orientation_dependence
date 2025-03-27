import argparse
import nibabel as nib
import numpy as np
import logging
import pandas as pd
from pathlib import Path

from scilpy.io.utils import add_verbose_arg

from modules.io import save_polyfits_as_npz
from modules.orientation_dependence import fit_single_fiber_results


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    
    p.add_argument('--in_bundles', nargs='+', default=[], required=True,
                   help='Characterization results for all bundles. \nShould '
                        'be the output of '
                        'scil_orientation_dependence_characterization.py. '
                        'All characterizations MUST have the same number of '
                        'bins.')
    
    p.add_argument('--in_bundles_names', nargs='+',
                   help='List of the names of the bundles, in the same order '
                        'as they were given. \nIf this argument is not used, '
                        'the script assumes that the name of the bundle \nis '
                        'the name of the parent folder.')

    p.add_argument('--reference', default="max-mean", type=str,
                   choices=["mean", "maximum", "max-mean"],
                   help='Choice of reference measure saved for later '
                        'correction. \nBy default, the weighted mean is taken '
                        'as the reference. \nTaking the maximum instead is '
                        'possible, but proven to be very noisy.')

    p.add_argument('--out_folder', default='orientation_dependence_plot.png',
                   help='Path and name of the output file.')

    p.add_argument('--min_nb_voxels', default=1, type=int,
                   help='Value of the minimal number of voxels per bin '
                        '[%(default)s].')

    p.add_argument('--check_nb_voxels_std', action='store_true',
                   help='If set, checks the std of nb_voxels along with '
                        'min_nb_voxels.')

    g2 = p.add_argument_group(title='Polyfit parameters')
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

    # Load all results
    nm_measures = []
    for name in ['MTR', 'ihMTR', 'MTsat', 'ihMTsat']:
        if name in list(np.load(args.in_bundles[0]).keys()):
            nm_measures.append(name)
    nb_measures = len(nm_measures)
    angles_min = np.load(args.in_bundles[0])['Angle_min']
    angles_max = np.load(args.in_bundles[0])['Angle_max']
    bins = np.concatenate((angles_min, [angles_max[-1]]))
    nb_bins = len(bins) - 1
    nb_bundles = len(args.in_bundles)
    bundles_names = np.empty((nb_bundles), dtype=object)
    measures = np.empty((nb_bundles, nb_bins, nb_measures))
    averages = np.zeros((nb_bundles, nb_measures))
    nb_voxels = np.empty((nb_bundles, nb_bins, nb_measures))
    nb_voxels_std = np.empty((nb_bundles, nb_bins, nb_measures))
    is_measures = np.empty((nb_bundles, nb_bins, nb_measures), dtype=bool)
    origins = np.empty((nb_bundles, nb_bins, nb_measures), dtype=object)
    for i, bundle in enumerate(args.in_bundles):
        # Verify that all subjects have the same number of bundles
        logging.info("Loading: {}".format(bundle))
        file = dict(np.load(bundle))
        if args.in_bundles_names:
            # Verify that number of bundle names equals number of bundles
            if len(args.in_bundles_names) != nb_bundles:
                parser.error("--in_bundles_names must contain the same "
                                "number of elements as in each set of "
                                "--in_bundles.")
            bundles_names[i] = args.in_bundles_names[i]
        else:
            bundles_names[i] = Path(bundle).parent.name
        for j, nm_measure in enumerate(nm_measures):
            measures[i, :, j] = file[nm_measure]
            nb_voxels[i, :, j] = file['Nb_voxels_' + nm_measure]
            if 'Nb_voxels_std_' + nm_measure in file.keys():
                nb_voxels_std[i, :, j] = file['Nb_voxels_std_' + nm_measure]
            is_measures[i, :, j] = nb_voxels[i, :, j] >= args.min_nb_voxels
            if args.check_nb_voxels_std:
                is_measures[i, :, j] = is_measures[i, :, j] & (nb_voxels_std[i, :, j] < nb_voxels[i, :, j]) & (nb_voxels_std[i, :, j] != 0)
            # Add check for empty is_measures!! If empty, maximum makes no sense.
            origins[i, :, j] = file['Origin_' + nm_measure]
            is_original = origins[i, :, j] == bundles_names[i]
            averages[i, j] = np.ma.average(np.ma.MaskedArray(measures[i, :, j][is_original],
                                                             mask=np.isnan(measures[i, :, j][is_original])),
                                                             weights=nb_voxels[i, :, j][is_original], axis=0)

    # For every bundle, fit the polynome
    for i, bundle_name in enumerate(bundles_names):
        logging.info("Fitting the results of bundle {}.".format(bundle_name))
        measures_fit, measures_max = fit_single_fiber_results(bins,
                                                              measures[i],
                                                              is_measures=is_measures[i],
                                                              nb_voxels=nb_voxels[i],
                                                              stop_crit=args.stop_crit,
                                                              use_weighted_polyfit=args.use_weighted_polyfit)
        out_path = Path(args.out_folder) / (bundles_names[i] + '/1f_polyfits')
        if args.reference == "mean":
            save_polyfits_as_npz(measures_fit, averages[i],
                                 nm_measures, out_path)
        elif args.reference == "maximum":
            save_polyfits_as_npz(measures_fit, measures_max,
                                 nm_measures, out_path)
        elif args.reference == "max-mean":
            save_polyfits_as_npz(measures_fit, measures_max - averages[i],
                                 nm_measures, out_path)


if __name__ == "__main__":
    main()
