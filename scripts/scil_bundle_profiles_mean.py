#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pro tip: There is no more --whole_wm argument. Simply pass the whole WM
characterization (and polyfit) as a bundle with name WM and put WM last in
--bundles_order.
"""

import argparse
import numpy as np
import logging
from pathlib import Path

from modules.io import save_profiles_mean_as_npz

from scilpy.io.utils import (add_verbose_arg, assert_inputs_exist,
                             assert_outputs_exist, add_overwrite_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    
    p.add_argument('--measures', nargs='+', required=True,
                   help='List of measures.')
    
    p.add_argument('--in_profiles', nargs='+', action='append', required=True,
                   help='')
    
    p.add_argument('--in_bundles_names', nargs='+',
                   help='List of the names of the bundles, in the same order '
                        'as they were given. \nIf this argument is not used, '
                        'the script assumes that the name of the bundle \nis '
                        'the name of the parent folder.')

    p.add_argument('--outdir', default='./',
                   help='Path to the output directory.')

    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    out_folder = Path(args.outdir)

    # This does not work since the inputs are lists of lists.
    # I would have to adjust assert_inputs_exist in scilpy
    # assert_inputs_exist(parser, args.in_bundles, args.in_polyfits)
    assert_outputs_exist(parser, args, args.outdir)

    # Load all results
    nm_measures = args.measures
    nb_measures = len(nm_measures)
    nb_subjects = len(args.in_profiles)
    nb_bundles = len(args.in_profiles[0])
    file_name = Path(args.in_profiles[0][0]).stem
    bundles_names = np.empty((nb_bundles), dtype=object)
    measures = np.empty((nb_measures, nb_subjects, nb_bundles, 3), dtype=object)
    measures_std = np.empty((nb_measures, nb_subjects, nb_bundles, 3),
                            dtype=object)
    nb_voxels = np.empty((nb_measures, nb_subjects, nb_bundles, 3), dtype=object)
    for i, sub in enumerate(args.in_profiles):
        # Verify that all subjects have the same number of bundles
        if len(sub) != nb_bundles:
            parser.error("Different sets of --in_bundles must have the same "
                         "number of bundles.")
        for j, bundle in enumerate(sub):
            logging.info("Loading: {}".format(bundle))
            file = dict(np.load(bundle))
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
                measures[k, i, j, 0] = file[nm_measure + '_af']
                measures_std[k, i, j, 0] = file[nm_measure + '_std_af']
                nb_voxels[k, i, j, 0] = file[nm_measure + '_voxel_count_af']
                measures[k, i, j, 1] = file[nm_measure + '_sf']
                measures_std[k, i, j, 1] = file[nm_measure + '_std_sf']
                nb_voxels[k, i, j, 1] = file[nm_measure + '_voxel_count_sf']
                measures[k, i, j, 2] = file[nm_measure + '_mf']
                measures_std[k, i, j, 2] = file[nm_measure + '_std_mf']
                nb_voxels[k, i, j, 2] = file[nm_measure + '_voxel_count_mf']

    mean_measures = np.empty((nb_measures, nb_bundles, 3), dtype=object)
    mean_measures_std = np.empty((nb_measures, nb_bundles, 3), dtype=object)
    mean_nb_voxels = np.empty((nb_measures, nb_bundles, 3), dtype=object)
    for j, bundle_name in enumerate(bundles_names):
        for k in range(nb_measures):
            mean_measures[k, j, 0] = np.mean(measures[k, :, j, 0], axis=0)
            mean_measures_std[k, j, 0] = np.std(measures[k, :, j, 0], axis=0)
            mean_nb_voxels[k, j, 0] = np.mean(nb_voxels[k, :, j, 0], axis=0)
            mean_measures[k, j, 1] = np.mean(measures[k, :, j, 1], axis=0)
            mean_measures_std[k, j, 1] = np.std(measures[k, :, j, 1], axis=0)
            mean_nb_voxels[k, j, 1] = np.mean(nb_voxels[k, :, j, 1], axis=0)
            mean_measures[k, j, 2] = np.mean(measures[k, :, j, 2], axis=0)
            mean_measures_std[k, j, 2] = np.std(measures[k, :, j, 2], axis=0)
            mean_nb_voxels[k, j, 2] = np.mean(nb_voxels[k, :, j, 2], axis=0)

    for j, bundle_name in enumerate(bundles_names):
        out_path = out_folder / (bundle_name + "/" + file_name)
        save_profiles_mean_as_npz(mean_measures[:, j, :],
                                  mean_measures_std[:, j, :],
                                  mean_nb_voxels[:, j, :], nm_measures, out_path)

if __name__ == "__main__":
    main()
