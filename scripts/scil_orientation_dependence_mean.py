#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import logging
from pathlib import Path

from modules.io import save_results_as_npz_mean

from scilpy.io.utils import (add_verbose_arg,
                             assert_outputs_exist, add_overwrite_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    
    p.add_argument('--measures', nargs='+', required=True,
                   help='List of measures.')
    
    p.add_argument('--in_bundles', nargs='+', action='append', required=True,
                   help='Characterization results for all bundles. \nShould '
                        'be the output of '
                        'scil_orientation_dependence_characterization.py. '
                        '\nFor a single set of results, this should be a list '
                        'of all bundles results. \nHowever, to plot multiple '
                        'sets of results, repeat the argument --in_bundles '
                        'for each set. \nFor instance with an original and a '
                        'corrected dataset: \n--in_bundles '
                        'original/*/results.npz --in_bundles '
                        'corrected/*/results.npz '
                        '\nMake sure that each set has the same number of '
                        'bundles.')
    
    p.add_argument('--in_bundles_names', nargs='+',
                   help='List of the names of the bundles, in the same order '
                        'as they were given. \nIf this argument is not used, '
                        'the script assumes that the name of the bundle \nis '
                        'the name of the parent folder.')

    p.add_argument('--outdir', default='./',
                   help='Path to the output directory.')
    
    p.add_argument('--reference', default="mean", type=str,
                   choices=["mean", "maximum"],
                   help='Choice of reference measure saved for later '
                        'correction. \nBy default, the weighted mean is taken '
                        'as the reference. \nTaking the maximum instead is '
                        'possible, but proven to be very noisy.')

    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    out_folder = Path(args.outdir)

    assert_outputs_exist(parser, args, args.outdir)

    # Load all results
    nm_measures = args.measures
    nb_measures = len(nm_measures)
    nb_subjects = len(args.in_bundles)
    nb_bundles = len(args.in_bundles[0])
    file_name = Path(args.in_bundles[0][0]).stem
    bundles_names = np.empty((nb_bundles), dtype=object)
    bins = np.empty((nb_bundles), dtype=object)
    measures = np.empty((nb_measures, nb_subjects, nb_bundles), dtype=object)
    measures_std = np.empty((nb_measures, nb_subjects, nb_bundles),
                            dtype=object)
    nb_voxels = np.empty((nb_measures, nb_subjects, nb_bundles), dtype=object)
    origins = np.empty((nb_measures, nb_subjects, nb_bundles), dtype=object)
    for i, sub in enumerate(args.in_bundles):
        # Verify that all subjects have the same number of bundles
        if len(sub) != nb_bundles:
            parser.error("Different sets of --in_bundles must have the same "
                         "number of bundles.")
        for j, bundle in enumerate(sub):
            logging.info("Loading: {}".format(bundle))
            file = dict(np.load(bundle))
            bins[j] = np.concatenate((file['Angle_min'],
                                      [file['Angle_max'][-1]]))
            if args.in_bundles_names:
                # Verify that number of bundle names equals number of bundles
                if len(args.in_bundles_names) != nb_bundles:
                    parser.error("--in_bundles_names must contain the same "
                                 "number of elements as in each set of "
                                 "--in_bundles.")
                bundle_name = args.in_bundles_names[j]
            else:
                bundle_name = Path(bundle).parent.name
            # Verify that all bundle names are the same between subjects
            if i > 0 and bundle_name != bundles_names[j]:
                parser.error("Bundles extracted from different sets do not "
                             "seem to be the same. Use --in_bundles_names if "
                             "the naming of files is not consistent across "
                             "sets.")
            else:
                bundles_names[j] = bundle_name
            for k, nm_measure in enumerate(nm_measures):
                measures[k, i, j] = file[nm_measure]
                if nm_measure + '_std' in file.keys():
                    measures_std[k, i, j] = file[nm_measure + '_std']
                nb_voxels[k, i, j] = file['Nb_voxels_' + nm_measure]
                origins[k, i, j] = file['Origin_' + nm_measure]
                nb_bins = len(measures[k, i, j])
                if nb_bins != len(measures[k, 0, j]):
                    parser.error("The number of bins for a given bundle and "
                                 "measure must be the same for all subjects.")

    mean_measures = np.empty((nb_measures, nb_bundles), dtype=object)
    mean_measures_std = np.empty((nb_measures, nb_bundles),
                                    dtype=object)
    mean_nb_voxels = np.empty((nb_measures, nb_bundles), dtype=object)
    mean_nb_voxels_std = np.empty((nb_measures, nb_bundles), dtype=object)
    mean_origins = np.empty((nb_measures, nb_bundles), dtype=object)
    for j, bundle_name in enumerate(bundles_names):
        for k in range(nb_measures):
            nb_bins = len(measures[k, 0, j])
            origin = np.array(list(origins[k, :, j])) == bundle_name
            measure = np.where(origin, np.array(list(measures[k, :, j])),
                                np.NaN)
            nb_voxel = np.where(origin, np.array(list(nb_voxels[k, :, j])),
                                np.NaN)
            mean_measures[k, j] = np.nanmean(measure, axis=0)
            mean_measures_std[k, j] = np.nanstd(measure, axis=0)
            mean_nb_voxels[k, j] = np.nanmean(nb_voxel, axis=0)
            mean_nb_voxels_std[k, j] = np.nanstd(nb_voxel, axis=0)
            mean_origins[k, j] = np.repeat(bundle_name, nb_bins)

    for j, bundle_name in enumerate(bundles_names):
        out_path = out_folder / (bundle_name + "/" + file_name)
        save_results_as_npz_mean(bins[j], mean_measures[:, j],
                                 mean_measures_std[:, j], mean_nb_voxels[:, j],
                                 mean_nb_voxels_std[:, j],
                                 mean_origins[:, j], nm_measures, out_path)

if __name__ == "__main__":
    main()
