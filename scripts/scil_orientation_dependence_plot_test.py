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

from scilpy.io.utils import add_verbose_arg

from modules.io import plot_init


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    
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
                        'its filename without extensions.')
    
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
    
    p.add_argument('--measures', nargs='+',
                   help='List of measures to plot.')

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

    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

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
            bundles_names.append(Path(bundle).name.split(".")[0])
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
        if all(name == all_bundles_names[0] for name in all_bundles_names):
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
        parser.error("Some bundles given in --bundles_order do not match"
                     "the names in --in_bundles_names or extracted from the"
                     "filenames.")
    # Compute max voxel count with only bundles in bundles_order
    max_count = 0
    for max_counts in all_max_counts:
        for bundle in bundles_order:
            bundle_idx = bundles_names.index(bundle)
            if max_counts[bundle_idx] > max_count:
                max_count = max_counts[bundle_idx]

    # Load all polyfits
    if args.polyfits:
        all_polyfits = []
        for set in args.polyfits:
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
        parser.error("Too many bundles and measures were given at the same"
                     "time. Try reducing the number of bundles to {} or less,"
                     "or the number of measures to {} or less."
                     .format(int(args.max_nb_bundles / 2),
                             int(args.max_nb_measures / 2)))
    if nb_bundles_to_plot > args.max_nb_bundles:
        parser.error("Too many bundles were given. Try reducing the number"
                     "of bundles below {}".format(args.max_nb_bundles + 1))
    if nb_measures > args.max_nb_measures:
        parser.error("Too many measures were given. Try reducing the number"
                     "of measures below {}".format(args.max_nb_measures + 1))

    # Compute the configuration of the plot
    if nb_measures > args.max_nb_measures / 2:
        nb_rows = nb_bundles_to_plot
        nb_columns = nb_measures
    else:
        nb_rows = int(np.ceil(nb_bundles_to_plot / 2))
        nb_columns = nb_measures * 2

    min_nb_voxels = args.min_nb_voxels

    # Set up the plot parameters. TODO clean this.
    plot_init(dims=(8, 8), font_size=10)
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['lines.linewidth'] = 0.5
    plt.rcParams['lines.markersize'] = 3
    plt.rcParams['axes.titlesize'] = 10

    # Put bins in the loop so that we can have various bin width in the same plot
    mid_bins = (results[0]['Angle_min'] + results[0]['Angle_max']) / 2.
    highres_bins = np.arange(0, 90 + 1, 0.5)

    fig, ax = plt.subplots(nb_rows, nb_columns, layout='constrained')
    for i, set in enumerate(sets):
        for j in range(nb_bundles_to_plot):
            col = j % 2 # Adjust this for both configs
            if col == 1:
                col = 2
            row = j // 2
            if bundles_order[j] in bundles_names:
                bundle_idx = bundles_names.index(bundles_order[j])
                result = set[bundle_idx]
                for k in range(2): # Adjust this for both configs
                    is_measures = result['Nb_voxels_' + measures[k]] >= min_nb_voxels
                    is_not_measures = np.invert(is_measures)
                    norm = mpl.colors.Normalize(vmin=0, vmax=max_count)
                    pts_origin = result['Origin_' + measures[k]]
                    is_original = pts_origin == bundles_names[j]
                    is_none = pts_origin == "None"
                    is_patched = np.logical_and(np.invert(is_none),
                                                np.invert(is_original))
                    colorbar = ax[row, col + k].scatter(mid_bins[is_original & is_measures],
                                                    result[measures[k]][is_original & is_measures],
                                                    c=result['Nb_voxels_' + measures[k]][is_original & is_measures],
                                                    cmap='Greys', norm=norm,
                                                    edgecolors=cm.naviaS(cmap_idx[k]), linewidths=1)
                    ax[row, col + k].scatter(mid_bins[is_original & is_not_measures],
                                        result[measures[k]][is_original & is_not_measures],
                                        c=result['Nb_voxels_' + measures[k]][is_original & is_not_measures],
                                        cmap='Greys', norm=norm, alpha=0.5,
                                        edgecolors=cm.naviaS(cmap_idx[k]), linewidths=1)
                    ax[row, col + k].scatter(mid_bins[is_patched & is_measures],
                                            result[measures[k]][is_patched & is_measures],
                                            c=result['Nb_voxels_' + measures[k]][is_patched & is_measures],
                                            cmap='Greys', norm=norm,
                                            edgecolors="red", linewidths=1)
                    ax[row, col + k].scatter(mid_bins[is_patched & is_not_measures],
                                        result[measures[k]][is_patched & is_not_measures],
                                        c=result['Nb_voxels_' + measures[k]][is_patched & is_not_measures],
                                        cmap='Greys', norm=norm, alpha=0.5,
                                        edgecolors="red", linewidths=1)

                if args.polyfits:
                    polynome_r = np.poly1d(polyfits[bundle_idx][measures[0] + "_polyfit"])
                    ax[row, col].plot(highres_bins, polynome_r(highres_bins), "--",
                                    color=cm.naviaS(cmap_idx[0]))
                    polynome_sat = np.poly1d(polyfits[bundle_idx][measures[1] + "_polyfit"])
                    ax[row, col + 1].plot(highres_bins, polynome_sat(highres_bins), "--",
                                    color=cm.naviaS(cmap_idx[1]))

                ax[row, col].set_ylim(0.975 * np.nanmin(result[measures[0]]),
                                    1.025 * np.nanmax(result[measures[0]]))
                ax[row, col].set_yticks([np.round(np.nanmin(result[measures[0]]), decimals=1),
                                        np.round(np.nanmax(result[measures[0]]), decimals=1)])
                ax[row, col].set_xlim(0, 90)
                # ax[row, col].tick_params(axis='y', labelcolor="C0")
                ax[row, col + 1].set_ylim(0.975 * np.nanmin(result[measures[1]]),
                                        1.025 * np.nanmax(result[measures[1]]))
                ax[row, col + 1].set_yticks([np.round(np.nanmin(result[measures[1]]), decimals=1),
                                            np.round(np.nanmax(result[measures[1]]), decimals=1)])
                ax[row, col + 1].set_xlim(0, 90)

                bundle_idx += 1
            else:
                is_measures = whole_wm['Nb_voxels'] >= min_nb_voxels
                is_not_measures = np.invert(is_measures)
                for k in range(2):
                    ax[row, col + k].scatter(whole_mid_bins[is_measures],
                                        whole_wm[measures[k]][is_measures],
                                        c=whole_wm['Nb_voxels'][is_measures],
                                        cmap='Greys', norm=norm,
                                        edgecolors=cm.naviaS(cmap_idx[k]), linewidths=1)
                    ax[row, col + k].scatter(whole_mid_bins[is_not_measures],
                                        whole_wm[measures[k]][is_not_measures],
                                        c=whole_wm['Nb_voxels'][is_not_measures],
                                        cmap='Greys', norm=norm, alpha=0.5,
                                        edgecolors=cm.naviaS(cmap_idx[k]), linewidths=1)

                    ax[row, col + k].set_ylim(0.975 * np.nanmin(whole_wm[measures[k]]),
                                        1.025 * np.nanmax(whole_wm[measures[k]]))
                    ax[row, col + k].set_yticks([np.round(np.nanmin(whole_wm[measures[k]]), decimals=1),
                                            np.round(np.nanmax(whole_wm[measures[k]]), decimals=1)])
                    ax[row, col + k].set_xlim(0, 90)

            ax[row, col + 1].yaxis.set_label_position("right")
            ax[row, col + 1].yaxis.tick_right()

            # if col == 0:
            #     ax[row, col + 1].legend(handles=[colorbar], labels=[bundles_names[i]],
            #                         loc='center left', bbox_to_anchor=(1.0, 0.5),
            #                         markerscale=0, handletextpad=-2.0, handlelength=2)
            # if col == 0:
            #     ax[row, col].legend(handles=[colorbar], labels=[bundles_names[i]],
            #                         loc='center left', bbox_to_anchor=(-0.6, 0.5),
            #                         markerscale=0, handletextpad=-2.0, handlelength=2)
            if row != nb_rows - 1:
                ax[row, col].get_xaxis().set_ticks([])
                ax[row, col + 1].get_xaxis().set_ticks([])

            if row == 0:
                ax[row, col].title.set_text(measures[0])
                ax[row, col + 1].title.set_text(measures[1])

            if bundles_names[j].split('_')[0] == 'SLF':
                fontsize = 7
            else:
                fontsize = 9
            ax[row, col].set_ylabel(bundles_names[j], labelpad=10, fontsize=fontsize)
            # if row == (nb_rows - 1) / 2 and col == 0:
            #     ax[row, col].set_ylabel('MTR', color="C0")
            #     ax[row, col + 1].set_ylabel('ihMTR', color="C2")
            #     ax[row, col].yaxis.set_label_coords(-0.2, 0.5)
            # if row == (nb_rows - 1) / 2 and col == 0:
            #     axt.set_ylabel('MTsat', color="C1")
            #     axt2.set_ylabel('ihMTsat', color="C4")
    fig.colorbar(colorbar, ax=ax[:, -1], location='right',
                 label="Voxel count", aspect=100)
    ax[nb_rows - 1, 0].set_xlabel(r'$\theta_a$')
    ax[nb_rows - 1, 0].set_xlim(0, 90)
    ax[nb_rows - 1, 0].set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax[nb_rows - 1, 1].set_xlabel(r'$\theta_a$')
    ax[nb_rows - 1, 1].set_xlim(0, 90)
    ax[nb_rows - 1, 1].set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax[nb_rows - 1, 2].set_xlabel(r'$\theta_a$')
    ax[nb_rows - 1, 2].set_xlim(0, 90)
    ax[nb_rows - 1, 2].set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax[nb_rows - 1, 3].set_xlabel(r'$\theta_a$')
    ax[nb_rows - 1, 3].set_xlim(0, 90)
    ax[nb_rows - 1, 3].set_xticks([0, 15, 30, 45, 60, 75, 90])
    # if nb_bundles % 2 != 0:
    #     ax[nb_rows - 1, 1].set_yticks([])
    fig.get_layout_engine().set(h_pad=0, hspace=0)
    if 'MTR' in measures:
        line = plt.Line2D([0.455, 0.455], [0.035,0.985], transform=fig.transFigure, color="black", linestyle=(0, (5, 5)), alpha=0.7)
    if 'ihMTR' in measures:
        line = plt.Line2D([0.458, 0.458], [0.035,0.985], transform=fig.transFigure, color="black", linestyle=(0, (5, 5)), alpha=0.7)
    fig.add_artist(line)
    # plt.show()
    plt.savefig(args.out_filename, dpi=500, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()