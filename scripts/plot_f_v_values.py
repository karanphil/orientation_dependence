#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pro tip: There is no more --whole_wm argument. Simply pass the whole WM
characterization (and polyfit) as a bundle with name WM and put WM last in
--bundles_order.
"""

import argparse
from cmcrameri import cm
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
    
    p.add_argument('--in_values_sf', nargs=2, required=True, action='append',
                   help='List of values to plot. Either F or V values')

    p.add_argument('--in_values_mf', nargs=2, action='append',
                   help='List of values to plot. Either F or V values')

    p.add_argument('--out_filename',
                   help='Path and name of the output file.')
    
    p.add_argument('--value_type', default="f",
                   choices=["f", "v"],
                   help='')

    g = p.add_argument_group(title="Plot parameters")

    g.add_argument("--figsize", default=[10, 4], nargs=2,
                   help='rcParams figure.figsize [%(default)s].')

    g.add_argument("--font_size", default=10, type=int,
                   help='rcParams font.size [%(default)s].')

    g.add_argument("--axes_labelsize", default=12, type=int,
                   help='rcParams axes.labelsize [%(default)s].')

    g.add_argument("--axes_titlesize", default=12, type=int,
                   help='rcParams axes.titlesize [%(default)s].')

    g.add_argument("--legend_fontsize", default=10, type=int,
                   help='rcParams legend.fontsize [%(default)s].')

    g.add_argument("--xtick_labelsize", default=10, type=int,
                   help='rcParams xtick.labelsize [%(default)s].')

    g.add_argument("--ytick_labelsize", default=10, type=int,
                   help='rcParams ytick.labelsize [%(default)s].')

    g.add_argument("--axes_linewidth", default=1, type=float,
                   help='rcParams axes.linewidth [%(default)s].')

    g.add_argument("--lines_linewidth", default=1.5, type=float,
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
    assert_outputs_exist(parser, args, args.out_filename)

    nb_sets = len(args.in_values_sf)
    is_mf_values = args.in_values_mf != None
    values_sf = np.empty((nb_sets,
                          4,np.loadtxt(args.in_values_sf[0][0],
                                       skiprows=1).shape[-1]))
    for i, in_values in enumerate(args.in_values_sf):
        for j, in_value in enumerate(in_values):
            values_sf[i, j*2: j*2+2] = np.loadtxt(in_value, skiprows=1)
    if is_mf_values:
        values_mf = np.empty((nb_sets,
                            4,np.loadtxt(args.in_values_mf[0][0],
                                        skiprows=1).shape[-1]))
        for i, in_values in enumerate(args.in_values_mf):
            for j, in_value in enumerate(in_values):
                values_mf[i, j*2: j*2+2] = np.loadtxt(in_value, skiprows=1)

    initialize_plot(args)

    # Set colormap
    if args.colormap:
        cmap = get_lookup_table(args.colormap)
        cmap_idx = np.arange(0, 100, 1)
    else:
        cmap = cm.naviaS
        cmap_idx = np.arange(2, 1000, 1)

    # plt.boxplot(values.swapaxes(0, 1))
    # plt.show()
    if args.value_type == "f":
        ylabel = "Relative flatness change (%)"
        ymin = -100
        ymax = 1100
    else:
        ylabel = "Relative variability change (%)"
        ymin = -30
        ymax = 130
    labels = []
    labels_list = ["max-mean", "maximum", "mean"]
    if is_mf_values:
        fig, ax = plt.subplots(1, 2, layout='constrained')
        ax[0].hlines(0, 0, 16, linestyles="dashed", colors="grey", alpha=0.5, linewidth=1)
        ax[1].hlines(0, 0, 16, linestyles="dashed", colors="grey", alpha=0.5, linewidth=1)
        for i in range(nb_sets):
            violin_parts = ax[0].violinplot(values_sf[i].swapaxes(0, 1), showmeans=True,
                                        positions=[i + 1, i + 5, i + 9, i + 13])#,
                                        #label=labels[i])
            for partname in ('cbars','cmins','cmaxes','cmeans'):
                vp = violin_parts[partname]
                if partname == 'cmedians':
                    vp.set_edgecolor('grey')
                else:
                    vp.set_edgecolor(cmap(cmap_idx[i]))
            for parts in violin_parts['bodies']:
                parts.set_facecolor(cmap(cmap_idx[i]))
                parts.set_edgecolor(cmap(cmap_idx[i]))
            labels.append((violin_parts['bodies'][0], labels_list[i]))
        ax[0].set_xticks(ticks=[2, 6, 10, 14], labels=["MTR", "MTsat", "ihMTR", "ihMTsat"])
        ax[0].set_ylabel(ylabel)
        #ax[0].set_ylim(max(-100, np.min(values_sf) - 5), min(100, np.max(values_sf) + 5))
        ax[0].set_ylim(ymin, ymax)
        ax[0].set_xlim(0, 16)
        if args.value_type == "f":
            ax[1].legend(*zip(*labels), loc=1, title="Reference")
        ax[0].set_title("Single-fiber voxels")
        for i in range(nb_sets):
            violin_parts = ax[1].violinplot(values_mf[i].swapaxes(0, 1), showmeans=True,
                                        positions=[i + 1, i + 5, i + 9, i + 13])#,
                                        #label=labels[i])
            for partname in ('cbars','cmins','cmaxes','cmeans'):
                vp = violin_parts[partname]
                if partname == 'cmedians':
                    vp.set_edgecolor('grey')
                else:
                    vp.set_edgecolor(cmap(cmap_idx[i]))
            for parts in violin_parts['bodies']:
                parts.set_facecolor(cmap(cmap_idx[i]))
                parts.set_edgecolor(cmap(cmap_idx[i]))
            labels.append((violin_parts['bodies'][0], labels_list[i]))
        ax[1].set_xticks(ticks=[2, 6, 10, 14], labels=["MTR", "MTsat", "ihMTR", "ihMTsat"])
        # ax[1].set_ylabel(ylabel)
        #ax[1].set_ylim(max(-100, np.min(values_sf) - 5), min(100, np.max(values_sf) + 5))
        ax[1].set_ylim(ymin, ymax)
        ax[1].set_xlim(0, 16)
        ax[1].set_title("Multi-fiber voxels")
        # plt.show()
        plt.savefig(args.out_filename, dpi=500, bbox_inches='tight')
        plt.close()
    else:
        fig, ax = plt.subplots(1, 1, layout='constrained')
        ax.hlines(0, 0, 16, linestyles="dashed", colors="grey", alpha=0.5, linewidth=1)
        for i in range(nb_sets):
            violin_parts = ax.violinplot(values_sf[i].swapaxes(0, 1), showmeans=True,
                                        positions=[i + 1, i + 5, i + 9, i + 13])#,
                                        #label=labels[i])
            for partname in ('cbars','cmins','cmaxes','cmeans'):
                vp = violin_parts[partname]
                if partname == 'cmedians':
                    vp.set_edgecolor('grey')
                else:
                    vp.set_edgecolor(cmap(cmap_idx[i]))
            for parts in violin_parts['bodies']:
                parts.set_facecolor(cmap(cmap_idx[i]))
                parts.set_edgecolor(cmap(cmap_idx[i]))
            labels.append((violin_parts['bodies'][0], labels_list[i]))
        ax.set_xticks(ticks=[2, 6, 10, 14], labels=["MTR", "MTsat", "ihMTR", "ihMTsat"])
        ax.set_ylabel(ylabel)
        #ax.set_ylim(max(-100, np.min(values_sf) - 5), min(100, np.max(values_sf) + 5))
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(0, 16)
        ax.legend(*zip(*labels), loc=1, title="Reference")
        # ax.set_title("Single-fiber voxels")
        # plt.show()
        plt.savefig(args.out_filename, dpi=500, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()
