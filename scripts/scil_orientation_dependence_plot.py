#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pro tip: There is no more --whole_wm argument. Simply pass the whole WM
characterization (and polyfit) as a bundle with name WM and put WM last in
--bundles_order.
"""

import argparse
from cmcrameri import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import logging
from pathlib import Path

from scilpy.io.utils import (add_verbose_arg, assert_inputs_exist,
                             assert_outputs_exist, add_overwrite_arg)
from scilpy.viz.color import get_lookup_table

from modules.io import initialize_plot


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
    
    p.add_argument('--bundles_order', nargs='+',
                   help='Order in which to plot the bundles. This does not '
                        'have to be the same length \nas --in_bundles_names. '
                        'Thus, it can be used to select only a few bundles. '
                        '\nBy default, will follow the order given by '
                        '--in_bundles.')
    
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

    p.add_argument('--out_filename', default='orientation_dependence_plot.png',
                   help='Path and name of the output file.')

    p.add_argument('--min_nb_voxels', default=30, type=int,
                   help='Value of the minimal number of voxels per bin '
                        '[%(default)s].')

    p.add_argument('--max_nb_bundles', default=34, type=int,
                   help='Maximum number of bundles that can be plotted '
                        '\nIt is not recommended to change this value '
                        '[%(default)s].')
    
    p.add_argument('--max_nb_measures', default=4, type=int,
                   help='Maximum number of measures that can be plotted. '
                        '\nIt is not recommended to change this value '
                        '[%(default)s].')
    
    p.add_argument('--plot_mean', action='store_true',
                   help='If set, the mean of all sets is plotted. Cannot be '
                        'set with in_polyfits.')

    p.add_argument('--plot_std', action='store_true',
                   help='If set, the std is plotted.')

    p.add_argument('--write_mean_std', action='store_true',
                   help='If set, the mean std is added as text.') # Update help

    p.add_argument('--horizontal_test', action='store_true',
                   help='If set, the ratio between...') # Update help

    p.add_argument('--norm_with_wm', action='store_true',
                   help='If set, the normalization...') # Update help

    p.add_argument('--common_yticks', action='store_true',
                   help='If set, all plots of the same measure will have the '
                        'same yticks.')

    p.add_argument('--set_yticks', nargs='+', action='append', type=float,
                   help='Given yticks per measure. For each measure, use the '
                        '--set_yticks argument to give all yticks. \nThe '
                        'minimum * 0.975 and maximum * 1.025 for each will be '
                        'used for ylim. \nMust be in the same order as '
                        '--measures.')

    g = p.add_argument_group(title="Plot parameters")

    g.add_argument("--figsize", default=[8, 8], nargs=2,
                   help='rcParams figure.figsize [%(default)s].')

    g.add_argument("--font_size", default=10, type=int,
                   help='rcParams font.size [%(default)s].')

    g.add_argument("--axes_labelsize", default=10, type=int,
                   help='rcParams axes.labelsize [%(default)s].')

    g.add_argument("--axes_titlesize", default=10, type=int,
                   help='rcParams axes.titlesize [%(default)s].')

    g.add_argument("--legend_fontsize", default=8, type=int,
                   help='rcParams legend.fontsize [%(default)s].')

    g.add_argument("--xtick_labelsize", default=8, type=int,
                   help='rcParams xtick.labelsize [%(default)s].')

    g.add_argument("--ytick_labelsize", default=8, type=int,
                   help='rcParams ytick.labelsize [%(default)s].')

    g.add_argument("--axes_linewidth", default=1, type=float,
                   help='rcParams axes.linewidth [%(default)s].')

    g.add_argument("--lines_linewidth", default=0.5, type=float,
                   help='rcParams lines.linewidth [%(default)s].')

    g.add_argument("--lines_markersize", default=3, type=float,
                   help='rcParams lines.markersize [%(default)s].')

    g.add_argument('--colormap',
                   help='Select the colormap for colored trk (dps/dpp). '
                        '\nUse two Matplotlib named color '
                        'separeted by a - to create your own colormap. '
                        '\nBy default, will use the naviaS colormap from the '
                        'cmcrameri library.')

    g.add_argument("--legend_names", nargs='+',
                   help='Names of the different sets of results, to be placed '
                        'in a legend located \nwith --legend_subplot and '
                        '--legend_location. \nIf not names are given, no '
                        'legend will be added.')

    g.add_argument("--legend_subplot", nargs=2,
                   help='String describing in which subplot to place the '
                        'legend, in the case of multiple sets. \nShould be the '
                        'name of one of the bundles in --bundles_order, '
                        '\nfollowed by the name of one of the measures in '
                        '--measures. For instance, CST_L MTR. \nBy default, '
                        'the legend will be placed in the last subplot.')

    g.add_argument("--legend_location", type=int,
                   help='Location of the legend inside the selected subplot. '
                        '\nBy default, the matplotlib will decide the '
                        'location inside the subplot.')

    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_bundles, args.in_polyfits)
    assert_outputs_exist(parser, args, args.out_filename)

    # if args.in_polyfits and args.plot_mean:
    #     parser.error('Polynomial fits cannot be given with the plot_mean '
    #                  'option.')

    if args.common_yticks and args.set_yticks:
        parser.error('Argument common_yticks and set_yticks cannot be used '
                     'together.')

    # Load all results
    nm_measures = args.measures
    nb_measures = len(nm_measures)
    nb_subjects = len(args.in_bundles)
    nb_bundles = len(args.in_bundles[0])
    bundles_names = np.empty((nb_bundles), dtype=object)
    angles_min = np.empty((nb_subjects, nb_bundles), dtype=object)
    angles_max = np.empty((nb_subjects, nb_bundles), dtype=object)
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
            angles_min[i, j] = file['Angle_min']
            angles_max[i, j] = file['Angle_max']
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
                if args.plot_mean and nb_bins != len(measures[k, 0, j]):
                    parser.error("When the plot_mean option is given, the "
                                 "number of bins for a given bundle and "
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

    if args.plot_mean:
        mean_measures = np.empty((nb_measures, 1, nb_bundles), dtype=object)
        mean_measures_std = np.empty((nb_measures, 1, nb_bundles),
                                     dtype=object)
        mean_nb_voxels = np.empty((nb_measures, 1, nb_bundles), dtype=object)
        mean_origins = np.empty((nb_measures, 1, nb_bundles), dtype=object)
        mean_polyfits = np.empty((nb_measures, 1, nb_bundles), dtype=object)
        for j, bundle_name in enumerate(bundles_names):
            for k in range(nb_measures):
                nb_bins = len(measures[k, 0, j])
                origin = np.array(list(origins[k, :, j])) == bundle_name
                measure = np.where(origin, np.array(list(measures[k, :, j])),
                                   np.NaN)
                nb_voxel = np.where(origin, np.array(list(nb_voxels[k, :, j])),
                                    np.NaN)
                mean_measures[k, 0, j] = np.nanmean(measure, axis=0)
                mean_measures_std[k, 0, j] = np.nanstd(measure, axis=0)
                mean_nb_voxels[k, 0, j] = np.nanmean(nb_voxel, axis=0)
                mean_origins[k, 0, j] = np.repeat(bundle_name, nb_bins)
                if args.in_polyfits:
                    polyfit = np.array(list(polyfits[k, :, j]))
                    mean_polyfits[k, 0, j] = np.mean(polyfit, axis=0)
        measures = mean_measures
        measures_std = mean_measures_std
        nb_voxels = mean_nb_voxels
        origins = mean_origins
        polyfits = mean_polyfits
        nb_subjects = 1

    if args.bundles_order:
        bundles_order = args.bundles_order
        # Verify that all bundles in bundles_order are present in bundles_names
        if not all(bundle in bundles_names for bundle in bundles_order):
            parser.error("Some bundles given in --bundles_order do not match "
                         "the names in --in_bundles_names or extracted from "
                         "the filenames.")
    else:
        bundles_order = bundles_names
    nb_bundles_to_plot = len(bundles_order)

    # Compute max voxel count with only bundles in bundles_order and
    # the minimal/maximal values for each measures
    max_count = 0
    ymin = np.ones((nb_measures)) * 1000000
    ymax = np.zeros((nb_measures))
    for bundle in bundles_order:
        bundle_idx = np.argwhere(bundles_names == bundle)[0][0]
        for i in range(nb_measures):
            for j in range(nb_subjects):
                if bundle == "WM" and args.norm_with_wm:
                    if np.nanmax(nb_voxels[i, j, bundle_idx]) > max_count:
                        max_count = np.nanmax(nb_voxels[i, j, bundle_idx])
                elif bundle != "WM":
                    if np.nanmax(nb_voxels[i, j, bundle_idx]) > max_count:
                        max_count = np.nanmax(nb_voxels[i, j, bundle_idx])                    
                if np.nanmax(measures[i, j, bundle_idx]) > ymax[i]:
                    ymax[i] = np.nanmax(measures[i, j, bundle_idx])
                if np.nanmin(measures[i, j, bundle_idx]) < ymin[i]:
                    ymin[i] = np.nanmin(measures[i, j, bundle_idx])
    norm = mpl.colors.Normalize(vmin=0, vmax=max_count)

    # Verify the dimensions of the plot
    if (nb_bundles_to_plot > args.max_nb_bundles / 2 and
        nb_measures > args.max_nb_measures / 2):
        parser.error("Too many bundles and measures were given at the same "
                     "time. Try reducing the number of bundles to {} or less, "
                     "or the number of measures to {} or less."
                     .format(int(args.max_nb_bundles / 2),
                             int(args.max_nb_measures / 2)))
    if nb_bundles_to_plot > args.max_nb_bundles:
        parser.error("Too many bundles were given. Try reducing the number "
                     "of bundles below {}".format(args.max_nb_bundles + 1))
    if nb_measures > args.max_nb_measures:
        parser.error("Too many measures were given. Try reducing the number "
                     "of measures below {}".format(args.max_nb_measures + 1))
        
    # Verify the legend subplot exists
    if args.legend_subplot:
        if (args.legend_subplot[0] not in bundles_order or
            args.legend_subplot[1] not in measures):
            parser.error("Given legend subplot does not exist. Make sure both "
                         "names are in either --bundles_order or --measures.")

    # Compute the configuration of the plot
    if nb_measures > args.max_nb_measures / 2:
        nb_rows = nb_bundles_to_plot
        nb_columns = nb_measures
        split_columns = False
    else:
        nb_rows = int(np.ceil(nb_bundles_to_plot / 2))
        nb_columns = nb_measures * 2
        split_columns = True

    min_nb_voxels = args.min_nb_voxels
    highres_bins = np.arange(0, 90 + 1, 0.5)
    # Set colormap
    if args.colormap:
        cmap = get_lookup_table(args.colormap)
        cmap_idx = np.arange(0, 100, 1)
    else:
        cmap = cm.naviaS
        cmap_idx = np.arange(2, 1000, 1)

    initialize_plot(args)
    fig, ax = plt.subplots(nb_rows, nb_columns, layout='constrained')
    min_measures = np.ones((nb_bundles_to_plot, nb_measures)) * 10000000
    max_measures = np.zeros((nb_bundles_to_plot, nb_measures))
    colorbars = np.empty((nb_subjects), dtype=object)
    for i in range(nb_subjects):
        for j in range(nb_bundles_to_plot):
            if split_columns:  # for nb_measures <= args.max_nb_measures / 2
                col = (j % 2) * int(nb_columns / 2)
                row = j // 2
            else:  # for nb_measures > args.max_nb_measures / 2
                col = 0
                row = j

            jj = np.argwhere(bundles_names == bundles_order[j])[0][0]
            mid_bins = (angles_min[i, jj] + angles_max[i, jj]) / 2.
            for k in range(nb_measures):
                is_measures = nb_voxels[k, i, jj] >= min_nb_voxels
                is_not_measures = np.invert(is_measures)
                color = cmap(cmap_idx[i + (nb_subjects == 1) * k])
                pts_origin = origins[k, i, jj]
                is_original = pts_origin == bundles_order[j]
                is_none = pts_origin == "None"
                is_patched = np.logical_and(np.invert(is_none),
                                            np.invert(is_original))
                cb = ax[row, col + k].scatter(mid_bins[is_original & is_measures],
                                              measures[k, i, jj][is_original & is_measures],
                                              c=nb_voxels[k, i, jj][is_original & is_measures],
                                              cmap='Greys', norm=norm,
                                              edgecolors=color,
                                              linewidths=1)
                ax[row, col + k].scatter(mid_bins[is_original & is_not_measures],
                                         measures[k, i, jj][is_original & is_not_measures],
                                         c=nb_voxels[k, i, jj][is_original & is_not_measures],
                                         cmap='Greys', norm=norm, alpha=0.5,
                                         edgecolors=color, linewidths=1)
                ax[row, col + k].scatter(mid_bins[is_patched & is_measures],
                                         measures[k, i, jj][is_patched & is_measures],
                                         c=nb_voxels[k, i, jj][is_patched & is_measures],
                                         cmap='Greys', norm=norm, marker="s",
                                         edgecolors=color, linewidths=1)
                ax[row, col + k].scatter(mid_bins[is_patched & is_not_measures],
                                         measures[k, i, jj][is_patched & is_not_measures],
                                         c=nb_voxels[k, i, jj][is_patched & is_not_measures],
                                         cmap='Greys', norm=norm, marker="s",
                                         alpha=0.5, edgecolors=color,
                                         linewidths=1)
                colorbars[i] = cb

                if args.horizontal_test: # This only works for 2 series of input!!! Modify this later.
                    is_not_nan = nb_voxels[k, 0, jj] >= 1
                    # polyfit1 = np.polyfit(mid_bins[is_not_nan],
                    #                       measures[k, 0, jj][is_not_nan],
                    #                       0,
                    #                       w=np.sqrt(nb_voxels[k, 0, jj][is_not_nan]),
                    #                       full=True)
                    # polyfit2 = np.polyfit(mid_bins[is_not_nan],
                    #                       measures[k, 1, jj][is_not_nan],
                    #                       0,
                    #                       w=np.sqrt(nb_voxels[k, 1, jj][is_not_nan]),
                    #                       full=True)
                    # res1 = np.sqrt(polyfit1[1][0])
                    # res2 = np.sqrt(polyfit2[1][0])
                    # STD pondérée
                    average1 = np.average(measures[k, 0, jj][is_not_nan],
                                          weights=np.sqrt(nb_voxels[k, 0, jj])[is_not_nan])
                    var1 = np.average((measures[k, 0, jj][is_not_nan] - average1)**2,
                                      weights=np.sqrt(nb_voxels[k, 0, jj])[is_not_nan])

                    average2 = np.average(measures[k, 1, jj][is_not_nan],
                                          weights=np.sqrt(nb_voxels[k, 1, jj])[is_not_nan])

                    var2 = np.average((measures[k, 1, jj][is_not_nan] - average2)**2,
                                      weights=np.sqrt(nb_voxels[k, 1, jj])[is_not_nan])
                    std1 = np.sqrt(var1)
                    std2 = np.sqrt(var2)
                    ax[row, col + k].text(0.01, 0.03,
                                          "F: " + str(np.round((std1 - std2) / std1 * 100, decimals=1)) + "%",
                                          color="dimgrey",
                                          transform=ax[row, col + k].transAxes,
                                          size=6)

                if args.in_polyfits:
                    polynome_r = np.poly1d(polyfits[k, i, jj])
                    ax[row, col + k].plot(highres_bins, polynome_r(highres_bins),
                                          "--", color=color)

                if args.plot_std:
                    ax[row, col + k].fill_between(mid_bins,
                                                  measures[k, i, jj] - measures_std[k, i, jj],
                                                  measures[k, i, jj] + measures_std[k, i, jj],
                                                  color=color,
                                                  edgecolor=None,
                                                  alpha=0.3)
                
                if args.write_mean_std: # This only works for 2 series of input!!! Modify this later.
                    mean1 = np.ma.average(np.ma.MaskedArray(measures[k, 0, jj],
                                                           mask=np.isnan(measures[k, 0, jj])),
                                                           weights=nb_voxels[k, 0, jj])
                    mean_std1 = np.ma.average(np.ma.MaskedArray(measures_std[k, 0, jj],
                                                               mask=np.isnan(measures[k, 0, jj])),
                                                               weights=nb_voxels[k, 0, jj])
                    
                    mean2 = np.ma.average(np.ma.MaskedArray(measures[k, 1, jj],
                                                           mask=np.isnan(measures[k, 1, jj])),
                                                           weights=nb_voxels[k, 1, jj])
                    mean_std2 = np.ma.average(np.ma.MaskedArray(measures_std[k, 1, jj],
                                                               mask=np.isnan(measures[k, 1, jj])),
                                                               weights=nb_voxels[k, 1, jj])
                    # Ratio
                    # ax[row, col + k].text(0.45, 0.1,
                    #                       str(np.round(((mean_std1 / mean1)/(mean_std2 / mean2)) * 100, decimals=1)) + "%",
                    #                       color="dimgrey",
                    #                       transform=ax[row, col + k].transAxes,
                    #                       size=6)
                    # Écart-relatif pondéré par moyenne
                    # ax[row, col + k].text(0.45, 0.1,
                    #                       str(np.round(((mean_std1 / mean1)/(mean_std2 / mean2) - 1) * 100, decimals=1)) + "%",
                    #                       color="dimgrey",
                    #                       transform=ax[row, col + k].transAxes,
                    #                       size=6)
                    # Écart-relatif
                    ax[row, col + k].text(0.70, 0.03,
                                          "V: " + str(np.round((mean_std1 - mean_std2) / mean_std1 * 100, decimals=1)) + "%",
                                          color="dimgrey",
                                          transform=ax[row, col + k].transAxes,
                                          size=6)

                if args.common_yticks:
                    ax[row, col + k].set_ylim(ymin[k] * 0.975, ymax[k] * 1.025)
                    ax[row, col + k].set_yticks([np.round(ymin[k], decimals=1),
                                                 np.round(ymax[k],
                                                          decimals=1)])
                elif args.set_yticks is not None:
                    yticks = args.set_yticks[k]
                    ax[row, col + k].set_ylim(np.min(yticks) * 0.975,
                                              np.max(yticks) * 1.025)
                    ax[row, col + k].set_yticks(yticks)
                else:
                    if 0.975 * np.nanmin(measures[k, i, jj]) < min_measures[j, k]:
                        min_measures[j, k] = 0.975 * np.nanmin(measures[k, i, jj])
                    if 1.025 * np.nanmax(measures[k, i, jj]) > max_measures[j, k]:
                        max_measures[j, k] = 1.025 * np.nanmax(measures[k, i, jj])
                    ax[row, col + k].set_ylim(min_measures[j, k],
                                              max_measures[j, k])
                    ax[row, col + k].set_yticks([np.round(min_measures[j, k],
                                                decimals=1),
                                                np.round(max_measures[j, k],
                                                decimals=1)])
                ax[row, col + k].set_xlim(0, 90)

                if (col + k) % 2 != 0:
                    ax[row, col + k].yaxis.set_label_position("right")
                    ax[row, col + k].yaxis.tick_right()

                if row != nb_rows - 1:
                    ax[row, col + k].get_xaxis().set_ticks([])
                else:
                    ax[row, col + k].set_xlabel(r'$\theta_a$')
                    ax[row, col + k].set_xlim(0, 90)
                    ax[row, col + k].set_xticks([0, 15, 30, 45, 60, 75, 90])

                if row == 0:
                    ax[row, col + k].title.set_text(nm_measures[k])

                if (args.legend_names and args.legend_subplot and
                    i == nb_subjects - 1):
                    if (args.legend_subplot[0] == bundles_order[j] and
                        args.legend_subplot[1] == nm_measures[k]):
                        ax[row, col + k].legend(handles=list(colorbars),
                                                labels=args.legend_names,
                                                loc=args.legend_location)

            if len(bundles_order[j]) > 6:
                fontsize = 7
            else:
                fontsize = 9
            ax[row, col].set_ylabel(bundles_order[j], labelpad=10,
                                    fontsize=fontsize)

    if args.legend_names and not args.legend_subplot:
        ax[row, col + k].legend(handles=list(colorbars),
                                labels=args.legend_names,
                                loc=args.legend_location)

    fig.colorbar(colorbars[0], ax=ax[:, -1], location='right',
                 label="Voxel count", aspect=100)
    fig.get_layout_engine().set(h_pad=0, hspace=0)
    plt.savefig(args.out_filename, dpi=500, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
