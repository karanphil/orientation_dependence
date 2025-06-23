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

    p.add_argument('--tractometry', action='store_true',
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
    if is_mf_values and not args.tractometry:
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
    elif is_mf_values and args.tractometry:
        symbols = ["o", "s", "^"]
        diffs = np.zeros((2,) + (values_mf.shape[1],) + (values_mf.shape[2],))
        diffs[0] = (values_mf[1] - values_mf[0]) / values_mf[0]
        diffs[1] = (values_mf[2] - values_mf[0]) / values_mf[0]
        diffs[:, 1, :] += 0.5
        diffs[:, 2, :] += 1
        diffs[:, 3, :] += 1.5
        fig, ax = plt.subplots(1, 2, layout='constrained')
        ax[0].hlines(0, 0, 16, linestyles="dashed", colors="grey", alpha=0.5, linewidth=1)
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
        ax[0].legend(*zip(*labels), loc=1, title="Reference")
        # ax[0].set_title("Track-profiles variability")
        ax[1].set_xlim(-1, 32 * 3 + 1)
        ax[1].hlines(0, -1, 32 * 3 + 1, linestyles="dashed", colors="grey", alpha=0.5, linewidth=1)
        ax[1].hlines(0.5, -1, 32 * 3 + 1, linestyles="dashed", colors="grey", alpha=0.5, linewidth=1)
        ax[1].hlines(1, -1, 32 * 3 + 1, linestyles="dashed", colors="grey", alpha=0.5, linewidth=1)
        ax[1].hlines(1.5, -1, 32 * 3 + 1, linestyles="dashed", colors="grey", alpha=0.5, linewidth=1)

        # for i in range(nb_sets):
        #     for j in range(values_mf[i].shape[-1]):
        #         ax[1].scatter([4*j + i, 4*j + i, 4*j + i, 4*j + i], values_mf[i, :, j], color=cmap(cmap_idx[j]), marker=symbols[i])
        # ax[1].set_yticks(ticks=[1, 1.5, 2, 2.5], labels=["MTR", "MTsat", "ihMTR", "ihMTsat"])
        # ax[1].set_xticks(ticks=np.arange(0, 33, 1) * 4 + 1, labels=["AF_L", "AF_R", "CC_1", "CC_2a", "CC_2b", "CC_3", "CC_4",
        #                                                     "CC_5", "CC_6", "CC_7", "CG_L", "CG_R", "CR_L", "CR_R",
        #                                                     "CST_L", "CST_R", "ICP_L", "ICP_R", "IFOF_L", "IFOF_R",
        #                                                     "ILF_L", "ILF_R", "OR_L", "OR_R", "SLF_1_L", "SLF_1_R",
        #                                                     "SLF_2_L", "SLF_2_R", "SLF_3_L", "SLF_3_R", "UF_L",
        #                                                     "UF_R", "MCP"], rotation='vertical', fontsize=6)
        for i in range(nb_sets - 1):
            for j in range(values_mf[i].shape[-1]):
                ax[1].scatter([3*j, 3*j, 3*j, 3*j], diffs[i, :, j], color=cmap(cmap_idx[j]), marker=symbols[i])
        ax[1].set_ylabel("Relative mean measures change")
        ax[1].yaxis.tick_right()
        ax[1].set_yticks(ticks=[0, 0.5, 1, 1.5], labels=["MTR", "MTsat", "ihMTR", "ihMTsat"], rotation='vertical', va='center')
        ax[1].set_xticks(ticks=np.arange(0, 33, 1) * 3, labels=["AF_L", "AF_R", "CC_1", "CC_2a", "CC_2b", "CC_3", "CC_4",
                                                            "CC_5", "CC_6", "CC_7", "CG_L", "CG_R", "CR_L", "CR_R",
                                                            "CST_L", "CST_R", "ICP_L", "ICP_R", "IFOF_L", "IFOF_R",
                                                            "ILF_L", "ILF_R", "OR_L", "OR_R", "SLF_1_L", "SLF_1_R",
                                                            "SLF_2_L", "SLF_2_R", "SLF_3_L", "SLF_3_R", "UF_L",
                                                            "UF_R", "MCP"], rotation='vertical', fontsize=6)
        from matplotlib.lines import Line2D
        point1 = Line2D([0], [0], label='max-mean', marker='o', markersize=3, markeredgecolor=cmap(cmap_idx[0]),
                        markerfacecolor=cmap(cmap_idx[0]), linestyle='')
        point2 = Line2D([0], [0], label='mean', marker='s', markersize=3, markeredgecolor=cmap(cmap_idx[0]),
                        markerfacecolor=cmap(cmap_idx[0]), linestyle='')
        ax[1].legend(handles=[point1, point2], title="Reference", loc=(0.6, 0.1), handletextpad=0.1)
        # ax[1].set_xlim(0, 16)
        # plt.show()
        # plt.subplots_adjust(wspace=2)
        plt.savefig(args.out_filename, dpi=500, bbox_inches='tight')
        plt.close()
    elif not is_mf_values and  args.tractometry:
        symbols = ["o", "s", "^"]
        diffs = np.zeros((2,) + (values_sf.shape[1],) + (values_sf.shape[2],))
        diffs[0] = (values_sf[1] - values_sf[0]) / values_sf[0]
        diffs[1] = (values_sf[2] - values_sf[0]) / values_sf[0]
        diffs[:, 1, :] += 0.5
        diffs[:, 2, :] += 1
        diffs[:, 3, :] += 1.5
        fig, ax = plt.subplots(1, 1, layout='constrained')
        ax.set_xlim(-1, 32 * 3 + 1)
        ax.hlines(0, -1, 32 * 3 + 1, linestyles="dashed", colors="grey", alpha=0.5, linewidth=1)
        ax.hlines(0.5, -1, 32 * 3 + 1, linestyles="dashed", colors="grey", alpha=0.5, linewidth=1)
        ax.hlines(1, -1, 32 * 3 + 1, linestyles="dashed", colors="grey", alpha=0.5, linewidth=1)
        ax.hlines(1.5, -1, 32 * 3 + 1, linestyles="dashed", colors="grey", alpha=0.5, linewidth=1)
        for i in range(nb_sets - 1):
            for j in range(values_sf[i].shape[-1]):
                ax.scatter([3*j, 3*j, 3*j, 3*j], diffs[i, :, j], color=cmap(cmap_idx[j]), marker=symbols[i])
        # ax.set_ylabel("Relative mean measures change")
        # ax.yaxis.tick_right()
        ax.set_yticks(ticks=[0, 0.5, 1, 1.5], labels=["MTR", "MTsat", "ihMTR", "ihMTsat"], rotation='vertical', va='center')
        ax.set_xticks(ticks=np.arange(0, 33, 1) * 3, labels=["AF_L", "AF_R", "CC_1", "CC_2a", "CC_2b", "CC_3", "CC_4",
                                                            "CC_5", "CC_6", "CC_7", "CG_L", "CG_R", "CR_L", "CR_R",
                                                            "CST_L", "CST_R", "ICP_L", "ICP_R", "IFOF_L", "IFOF_R",
                                                            "ILF_L", "ILF_R", "OR_L", "OR_R", "SLF_1_L", "SLF_1_R",
                                                            "SLF_2_L", "SLF_2_R", "SLF_3_L", "SLF_3_R", "UR_L",
                                                            "UR_R", "MCP"], rotation='vertical', fontsize=6)
        from matplotlib.lines import Line2D
        point1 = Line2D([0], [0], label='max-mean', marker='o', markersize=3, markeredgecolor=cmap(cmap_idx[0]),
                        markerfacecolor=cmap(cmap_idx[0]), linestyle='')
        point2 = Line2D([0], [0], label='mean', marker='s', markersize=3, markeredgecolor=cmap(cmap_idx[0]),
                        markerfacecolor=cmap(cmap_idx[0]), linestyle='')
        ax.legend(handles=[point1, point2], title="Reference", loc=(0.74, 0.1), handletextpad=0.1)
        # plt.show()
        # plt.subplots_adjust(wspace=2)
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
