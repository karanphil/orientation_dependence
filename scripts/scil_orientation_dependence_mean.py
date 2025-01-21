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

from modules.io import (save_results_as_npz_mean, save_polyfits_as_npz_mean)

from scilpy.io.utils import (add_verbose_arg, assert_inputs_exist,
                             assert_outputs_exist, add_overwrite_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    
    p.add_argument('--measures', nargs='+', required=True,
                   help='List of measures to plot.')
    
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

    p.add_argument('--in_polyfits', nargs='+', action='append',
                   help='Polyfits results for all bundles. \nShould '
                        'be the output of '
                        'scil_orientation_dependence_characterization.py. '
                        '\nIf multiple sets were given for --in_bundles, '
                        'there should be an equal number of sets for this '
                        'argument too. \nFor instance with an '
                        'original and a corrected dataset: \n--in_polyfits '
                        'original/*/polyfits.npz --in_polyfits '
                        'corrected/*/polyfits.npz '
                        '\nMake sure that each set has the same number of '
                        'bundles.')

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

    # This does not work since the inputs are lists of lists.
    # I would have to adjust assert_inputs_exist in scilpy
    # assert_inputs_exist(parser, args.in_bundles, args.in_polyfits)
    assert_outputs_exist(parser, args, args.outdir)

    # Load all results
    nm_measures = args.measures
    nb_measures = len(nm_measures)
    nb_subjects = len(args.in_bundles)
    nb_bundles = len(args.in_bundles[0])
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

    # Load all polyfits
    if args.in_polyfits:
        polyfits = np.empty((nb_measures, nb_subjects, nb_bundles),
                            dtype=object)
        # Verify that polyfits and bundles have same number of sets
        if len(args.in_polyfits) != nb_subjects:
            parser.error("--in_polyfits must contain the same number of "
                         "sets as --in_bundles.")
        for i, sub in enumerate(args.in_polyfits):
            # Verify that number of polyfits equals number of bundles
            if len(sub) != nb_bundles:
                parser.error("--in_polyfits must contain the same number of "
                             "elements as in each set of --in_bundles.")
            for j, bundle in enumerate(sub):
                logging.info("Loading: {}".format(bundle))
                file = dict(np.load(bundle))
                for k, nm_measure in enumerate(nm_measures):
                    polyfits[k, i, j] = file[nm_measure + "_polyfit"]

    mean_measures = np.empty((nb_measures, nb_bundles), dtype=object)
    mean_measures_std = np.empty((nb_measures, nb_bundles),
                                    dtype=object)
    mean_nb_voxels = np.empty((nb_measures, nb_bundles), dtype=object)
    mean_origins = np.empty((nb_measures, nb_bundles), dtype=object)
    mean_polyfits = np.empty((nb_measures, nb_bundles), dtype=object)
    polyfits = np.empty((nb_measures, nb_bundles), dtype=object)
    mean_reference = np.empty((nb_measures, nb_bundles))
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
            mean_origins[k, j] = np.repeat(bundle_name, nb_bins)
            if args.in_polyfits:
                polyfit = np.array(list(polyfits[k, :, j]))
                mean_polyfits[k, j] = np.mean(polyfit, axis=0)
                if args.reference == "mean":
                    mean_reference[k, j] = np.ma.average(np.ma.MaskedArray(mean_measures[k, j],
                                                         mask=np.isnan(mean_measures[k, j])),
                                                         weights=mean_nb_voxels[k, j])
                elif args.reference == "maximum":
                    mean_reference[k, j] = np.nanmax(mean_measures[k, j])

    for j, bundle_name in enumerate(bundles_names):
        out_path = out_folder / (bundle_name + '/1f_results')
        save_results_as_npz_mean(bins[j], mean_measures[:, j],
                            mean_measures_std[:, j], mean_nb_voxels[:, j],
                            mean_origins[:, j], nm_measures, out_path)
        if args.in_polyfits:
            out_path = out_folder / (bundle_name + '/1f_polyfits')
            save_polyfits_as_npz_mean(mean_polyfits[:, j], mean_reference[:, j],
                                      nm_measures, out_path)


if __name__ == "__main__":
    main()
