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

    # This does not work since the inputs are lists of lists.
    # I would have to adjust assert_inputs_exist in scilpy
    # assert_inputs_exist(parser, args.in_bundles, args.in_polyfits)
    assert_outputs_exist(parser, args, args.out_filename)

    measures = args.measures
    nb_measures = len(measures)

    # Load all results
    sets = []
    all_nb_bundles = []
    all_bundles_names = []
    all_max_counts = []
    for set in args.in_bundles:
        bundles = []
        bundles_names = []
        max_counts = []
        for bundle in set:
            logging.info("Loading: {}".format(bundle))
            result = np.load(bundle)
            bundles.append(result)
            bundles_names.append(Path(bundle).parent.name)
            max_count = 0
            for measure in measures:
                curr_max_count = np.max(result['Nb_voxels_' + measure])
                if curr_max_count > max_count:
                    max_count = curr_max_count
            max_counts.append(max_count)
        sets.append(bundles)
        all_nb_bundles.append(len(set))
        all_bundles_names.append(bundles_names)
        all_max_counts.append(max_counts)
    nb_sets = len(sets)

    # Verify that all sets have the same dimension
    if all(nb_bundles == all_nb_bundles[0] for nb_bundles in all_nb_bundles):
        nb_bundles = all_nb_bundles[0]
    else:
        parser.error("Different sets of --in_bundles must have the same "
                     "number of bundles.")

    if not args.in_bundles_names:
        # Verify that all extracted bundle names are the same between sets
        if not all(name == all_bundles_names[0] for name in all_bundles_names):
            parser.error("Bundles extracted from different sets do not seem "
                         "to be the same. Use --in_bundles_names if the "
                         "naming of files is not consistent across sets.")
        bundles_names = all_bundles_names[0]
    else:
        # Verify that dimension of sets equals dimension of bundles names
        if len(args.in_bundles_names) != nb_bundles:
            parser.error("--in_bundles_names must contain the same number of "
                         "elements as in each set of --in_bundles.")
        bundles_names = args.in_bundles_names

    bundles_order = args.bundles_order
    nb_bundles_to_plot = len(bundles_order)
    # Verify that all bundles in bundles_order are present in bundles_names
    if not all(bundle in bundles_names for bundle in bundles_order):
        parser.error("Some bundles given in --bundles_order do not match "
                     "the names in --in_bundles_names or extracted from the "
                     "filenames.")
    # Compute max voxel count with only bundles in bundles_order
    max_count = 0
    for max_counts in all_max_counts:
        for bundle in bundles_order:
            bundle_idx = bundles_names.index(bundle)
            if max_counts[bundle_idx] > max_count:
                max_count = max_counts[bundle_idx]
    norm = mpl.colors.Normalize(vmin=0, vmax=max_count)

    # Load all polyfits
    if args.in_polyfits:
        all_polyfits = []
        for set in args.in_polyfits:
            polyfits = []
            for polyfit in set:
                logging.info("Loading: {}".format(polyfit))
                polyfits.append(np.load(polyfit))
            all_polyfits.append(polyfits)

            # Verify that number of polyfits equals number of bundles
            if len(polyfits) != nb_bundles:
                parser.error("--in_polyfits must contain the same number of "
                             "elements as in each set of --in_bundles.")
        # Verify that polyfits and bundles have same number of sets
        if len(all_polyfits) != nb_sets:
            parser.error("--in_polyfits must contain the same number of "
                         "sets as --in_bundles.")

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
        cmap_idx = np.arange(0, 18, 1)
    else:
        cmap = cm.naviaS
        cmap_idx = np.arange(2, 20, 1)

    initialize_plot(args)
    fig, ax = plt.subplots(nb_rows, nb_columns, layout='constrained')
    min_measures = np.ones((nb_bundles_to_plot, nb_measures)) * 10000000
    max_measures = np.zeros((nb_bundles_to_plot, nb_measures))
    colorbars = np.empty((nb_sets), dtype=object)
    for i, set in enumerate(sets):
        for j in range(nb_bundles_to_plot):
            if split_columns:  # for nb_measures <= args.max_nb_measures / 2
                col = (j % 2) * int(nb_columns / 2)
                row = j // 2
            else:  # for nb_measures > args.max_nb_measures / 2
                col = 0
                row = j

            bundle_idx = bundles_names.index(bundles_order[j])
            result = set[bundle_idx]
            mid_bins = (result['Angle_min'] + result['Angle_max']) / 2.
            for k in range(nb_measures):
                is_measures = result['Nb_voxels_' + measures[k]] >= min_nb_voxels
                is_not_measures = np.invert(is_measures)
                color = cmap(cmap_idx[i + (nb_sets == 1) * k])
                pts_origin = result['Origin_' + measures[k]]
                is_original = pts_origin == bundles_order[j]
                is_none = pts_origin == "None"
                is_patched = np.logical_and(np.invert(is_none),
                                            np.invert(is_original))
                cb = ax[row, col + k].scatter(mid_bins[is_original & is_measures],
                                              result[measures[k]][is_original & is_measures],
                                              c=result['Nb_voxels_' + measures[k]][is_original & is_measures],
                                              cmap='Greys', norm=norm,
                                              edgecolors=color,
                                              linewidths=1)
                ax[row, col + k].scatter(mid_bins[is_original & is_not_measures],
                                         result[measures[k]][is_original & is_not_measures],
                                         c=result['Nb_voxels_' + measures[k]][is_original & is_not_measures],
                                         cmap='Greys', norm=norm, alpha=0.5,
                                         edgecolors=color, linewidths=1)
                ax[row, col + k].scatter(mid_bins[is_patched & is_measures],
                                         result[measures[k]][is_patched & is_measures],
                                         c=result['Nb_voxels_' + measures[k]][is_patched & is_measures],
                                         cmap='Greys', norm=norm, marker="s",
                                         edgecolors=color, linewidths=1)
                ax[row, col + k].scatter(mid_bins[is_patched & is_not_measures],
                                         result[measures[k]][is_patched & is_not_measures],
                                         c=result['Nb_voxels_' + measures[k]][is_patched & is_not_measures],
                                         cmap='Greys', norm=norm, marker="s",
                                         alpha=0.5, edgecolors=color,
                                         linewidths=1)
                colorbars[i] = cb

                if args.in_polyfits:
                    polyfits = all_polyfits[i]
                    polynome_r = np.poly1d(polyfits[bundle_idx][measures[k] + "_polyfit"])
                    ax[row, col + k].plot(highres_bins, polynome_r(highres_bins),
                                          "--", color=color)

                if 0.975 * np.nanmin(result[measures[k]]) < min_measures[j, k]:
                    min_measures[j, k] = 0.975 * np.nanmin(result[measures[k]])
                if 1.025 * np.nanmax(result[measures[k]]) > max_measures[j, k]:
                    max_measures[j, k] = 1.025 * np.nanmax(result[measures[k]])
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
                    ax[row, col + k].title.set_text(measures[k])

                if (args.legend_names and args.legend_subplot and
                    i == nb_sets - 1):
                    if (args.legend_subplot[0] == bundles_order[j] and
                        args.legend_subplot[1] == measures[k]):
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
