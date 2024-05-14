import argparse
from cmcrameri import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from pathlib import Path
from scipy.interpolate import splrep, BSpline

from modules.io import plot_init


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('out_folder',
                   help='Path of the output folder for txt, png, masks and '
                        'measures.')
    
    p.add_argument('--results_1f', nargs='+', default=[],
                   action='append', required=True,
                   help='List of characterization results.')
    p.add_argument('--results_2f', nargs='+', default=[],
                   action='append', required=True,
                   help='List of characterization results.')

    p.add_argument('--bundles_names', nargs='+', default=[], action='append',
                   required=True,
                   help='List of names for the characterized bundles.')

    p.add_argument('--whole_WM', default=[],
                   help='Path to the whole WM characterization.')
    
    p.add_argument('--out_name', default='toto.png',
                   help='Name of the output file.')
    
    p.add_argument('--MT_corr')

    p.add_argument('--ihMT_corr')

    g = p.add_argument_group(title='Characterization parameters')
    g.add_argument('--min_nb_voxels', default=30, type=int,
                   help='Value of the minimal number of voxels per bin '
                        '[%(default)s].')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    out_folder = Path(args.out_folder)
    min_nb_voxels = args.min_nb_voxels

    bundles_names = args.bundles_names[0]

    results_1f = []
    results_2f = []
    extracted_bundles = []
    max_count = 0
    tmp = 0
    for i, (result_1f, result_2f) in enumerate(zip(args.results_1f[0], args.results_2f[0])):
        if str(Path(result_1f).parent) in bundles_names:
            print("Loading: ", result_1f, result_2f)
            results_1f.append(np.load(result_1f))
            results_2f.append(np.load(result_2f))
            extracted_bundles.append(str(Path(result_1f).parent))
            curr_max_count = np.max([np.max(results_1f[tmp]['Nb_voxels']),
                                     np.max(results_2f[tmp]['Nb_voxels'])])
            if curr_max_count > max_count:
                max_count = curr_max_count
            tmp += 1

    if args.whole_WM:
        print("Loading: ", args.whole_WM)
        whole_wm = np.load(args.whole_WM)
        whole_mid_bins = (whole_wm['Angle_min'] + whole_wm['Angle_max']) / 2.

    if "MCP" in bundles_names:
        bundles_names.remove("MCP")
        bundles_names.append("MCP")

    bundles_names.append("WM")

    mt_corr = None
    ihmt_corr = None
    if args.MT_corr:
        mt_corr = np.loadtxt(args.MT_corr)
    if args.ihMT_corr:
        ihmt_corr = np.loadtxt(args.ihMT_corr)

    nb_bundles = len(bundles_names)
    # nb_rows = int(np.ceil(nb_bundles / 2))
    nb_rows = nb_bundles

    highres_bins = np.arange(0, 90 + 1, 0.5)

    s_mtr = np.ones(nb_bundles) * 0.0005
    s_mtr[2] = 0.001
    s_mtr[5] = 0.0015
    s_mtr[6] = 0.0002
    s_mtr[7] = 0.00005

    s_mtsat = np.ones(nb_bundles) * 0.0005
    s_mtsat[0] = 0.00001
    s_mtsat[1] = 0.0001
    s_mtsat[2] = 0.0001
    s_mtsat[3] = 0.00005
    s_mtsat[5] = 0.0001
    s_mtsat[6] = 0.00001 
    s_mtsat[8] = 0.00001
    s_mtsat[11] = 0.00005

    s_ihmtr = np.ones(nb_bundles) * 0.0005
    s_ihmtr[2] = 0.001
    s_ihmtr[5] = 0.003
    s_ihmtr[10] = 0.00005

    s_ihmtsat = np.ones(nb_bundles) * 0.0005
    s_ihmtsat[3] = 0.00001
    s_ihmtsat[5] = 0.00005
    s_ihmtsat[7] = 0.000003
    s_ihmtsat[10] = 0.000001

    # out_path = out_folder / str("all_bundles_original_1f_LABELS.png")
    # out_path = out_folder / str("all_bundles_original_1f.png")
    out_path1 = out_folder / args.out_name
    plot_init(dims=(8, 8), font_size=10)
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['lines.linewidth'] = 0.5
    plt.rcParams['lines.markersize'] = 3
    plt.rcParams['axes.titlesize'] = 10
    fig, ax = plt.subplots(nb_rows, 4, layout='constrained')
    for i in range(nb_bundles):
        col = 0
        row = i
        if bundles_names[i] in extracted_bundles:
            for j in range (5):
                bundle_idx = extracted_bundles.index(bundles_names[i])
                if j == 4: ########## 1f #########
                    result = results_1f[bundle_idx]
                    mid_bins = (result['Angle_min'] + result['Angle_max']) / 2.
                    is_measures = result['Nb_voxels'] >= min_nb_voxels
                    is_not_measures = np.invert(is_measures)
                    norm = mpl.colors.Normalize(vmin=0, vmax=max_count)
                    colorbar = ax[row, col].scatter(mid_bins[is_measures],
                                                    result['MTR'][is_measures],
                                                    c=result['Nb_voxels'][is_measures],
                                                    cmap='Greys', norm=norm,
                                                    edgecolors=cm.naviaS(2), linewidths=1)
                    ax[row, col].scatter(mid_bins[is_not_measures],
                                        result['MTR'][is_not_measures],
                                        c=result['Nb_voxels'][is_not_measures],
                                        cmap='Greys', norm=norm, alpha=0.5,
                                        edgecolors=cm.naviaS(2), linewidths=1)
                    ax[row, col + 1].scatter(mid_bins[is_measures],
                                                    result['MTsat'][is_measures],
                                                    c=result['Nb_voxels'][is_measures],
                                                    cmap='Greys', norm=norm,
                                                    edgecolors=cm.naviaS(2), linewidths=1)
                    ax[row, col + 1].scatter(mid_bins[is_not_measures],
                                        result['MTsat'][is_not_measures],
                                        c=result['Nb_voxels'][is_not_measures],
                                        cmap='Greys', norm=norm, alpha=0.5,
                                        edgecolors=cm.naviaS(2), linewidths=1)
                    ax[row, col + 2].scatter(mid_bins[is_measures],
                                                    result['ihMTR'][is_measures],
                                                    c=result['Nb_voxels'][is_measures],
                                                    cmap='Greys', norm=norm,
                                                    edgecolors=cm.naviaS(2), linewidths=1)
                    ax[row, col + 2].scatter(mid_bins[is_not_measures],
                                        result['ihMTR'][is_not_measures],
                                        c=result['Nb_voxels'][is_not_measures],
                                        cmap='Greys', norm=norm, alpha=0.5,
                                        edgecolors=cm.naviaS(2), linewidths=1)
                    ax[row, col + 3].scatter(mid_bins[is_measures],
                                                    result['ihMTsat'][is_measures],
                                                    c=result['Nb_voxels'][is_measures],
                                                    cmap='Greys', norm=norm,
                                                    edgecolors=cm.naviaS(2), linewidths=1)
                    ax[row, col + 3].scatter(mid_bins[is_not_measures],
                                        result['ihMTsat'][is_not_measures],
                                        c=result['Nb_voxels'][is_not_measures],
                                        cmap='Greys', norm=norm, alpha=0.5,
                                        edgecolors=cm.naviaS(2), linewidths=1)

                    is_not_nan = result['Nb_voxels'] > 0
                    weights = np.sqrt(result['Nb_voxels'][is_not_nan]) / np.max(result['Nb_voxels'][is_not_nan])
                    mtr_fit = splrep(mid_bins[is_not_nan], result['MTR'][is_not_nan], w=weights, s=s_mtr[i])
                    ax[row, col].plot(highres_bins, BSpline(*mtr_fit)(highres_bins), "--", color=cm.naviaS(2))
                    mtsat_fit = splrep(mid_bins[is_not_nan], result['MTsat'][is_not_nan], w=weights, s=s_mtsat[i])
                    ax[row, col + 1].plot(highres_bins, BSpline(*mtsat_fit)(highres_bins), "--", color=cm.naviaS(2))
                    ihmtr_fit = splrep(mid_bins[is_not_nan], result['ihMTR'][is_not_nan], w=weights, s=s_ihmtr[i])
                    ax[row, col + 2].plot(highres_bins, BSpline(*ihmtr_fit)(highres_bins), "--", color=cm.naviaS(2))
                    ihmtsat_fit = splrep(mid_bins[is_not_nan], result['ihMTsat'][is_not_nan], w=weights, s=s_ihmtsat[i])
                    ax[row, col + 3].plot(highres_bins, BSpline(*ihmtsat_fit)(highres_bins), "--", color=cm.naviaS(2))

                    ax[row, col].set_ylim(0.975 * np.nanmin(result['MTR']),
                                        1.025 * np.nanmax(result['MTR']))
                    ax[row, col].set_yticks([np.round(np.nanmin(result['MTR']), decimals=1),
                                            np.round(np.nanmax(result['MTR']), decimals=1)])
                    ax[row, col].set_xlim(0, 90)
                    # ax[row, col].tick_params(axis='y', labelcolor="C0")
                    ax[row, col + 1].set_ylim(0.975 * np.nanmin(result['MTsat']),
                                            1.025 * np.nanmax(result['MTsat']))
                    ax[row, col + 1].set_yticks([np.round(np.nanmin(result['MTsat']), decimals=1),
                                                np.round(np.nanmax(result['MTsat']), decimals=1)])
                    ax[row, col + 1].set_xlim(0, 90)
                    # ax[row, col + 1].tick_params(axis='y', labelcolor="C2")
                    ax[row, col + 2].set_ylim(0.975 * np.nanmin(result['ihMTR']),
                                            1.025 * np.nanmax(result['ihMTR']))
                    ax[row, col + 2].set_yticks([np.round(np.nanmin(result['ihMTR']), decimals=1),
                                                np.round(np.nanmax(result['ihMTR']), decimals=1)])
                    ax[row, col + 2].set_xlim(0, 90)
                    ax[row, col + 3].set_ylim(0.975 * np.nanmin(result['ihMTsat']),
                                            1.025 * np.nanmax(result['ihMTsat']))
                    ax[row, col + 3].set_yticks([np.round(np.nanmin(result['ihMTsat']), decimals=1),
                                                np.round(np.nanmax(result['ihMTsat']), decimals=1)])
                    ax[row, col + 3].set_xlim(0, 90)

                    bundle_idx += 1
                else: ########## 2f #########
                    result = results_2f[bundle_idx]
                    mid_bins = (result['Angle_min'] + result['Angle_max']) / 2.
                    is_measures = np.diagonal(result['Nb_voxels'][j]) >= min_nb_voxels
                    is_not_measures = np.invert(is_measures)
                    norm = mpl.colors.Normalize(vmin=0, vmax=max_count)
                    colorbar = ax[row, col].scatter(mid_bins[is_measures],
                                                    np.diagonal(result['MTR'][j])[is_measures],
                                                    c=np.diagonal(result['Nb_voxels'][j])[is_measures],
                                                    cmap='Greys', norm=norm,
                                                    edgecolors=cm.naviaS(2+1+j), linewidths=1)
                    ax[row, col].scatter(mid_bins[is_not_measures],
                                        np.diagonal(result['MTR'][j])[is_not_measures],
                                        c=np.diagonal(result['Nb_voxels'][j])[is_not_measures],
                                        cmap='Greys', norm=norm, alpha=0.5,
                                        edgecolors=cm.naviaS(2+1+j), linewidths=1)
                    ax[row, col + 1].scatter(mid_bins[is_measures],
                                                    np.diagonal(result['MTsat'][j])[is_measures],
                                                    c=np.diagonal(result['Nb_voxels'][j])[is_measures],
                                                    cmap='Greys', norm=norm,
                                                    edgecolors=cm.naviaS(2+1+j), linewidths=1)
                    ax[row, col + 1].scatter(mid_bins[is_not_measures],
                                        np.diagonal(result['MTsat'][j])[is_not_measures],
                                        c=np.diagonal(result['Nb_voxels'][j])[is_not_measures],
                                        cmap='Greys', norm=norm, alpha=0.5,
                                        edgecolors=cm.naviaS(2+1+j), linewidths=1)
                    ax[row, col + 2].scatter(mid_bins[is_measures],
                                                    np.diagonal(result['ihMTR'][j])[is_measures],
                                                    c=np.diagonal(result['Nb_voxels'][j])[is_measures],
                                                    cmap='Greys', norm=norm,
                                                    edgecolors=cm.naviaS(2+1+j), linewidths=1)
                    ax[row, col + 2].scatter(mid_bins[is_not_measures],
                                        np.diagonal(result['ihMTR'][j])[is_not_measures],
                                        c=np.diagonal(result['Nb_voxels'][j])[is_not_measures],
                                        cmap='Greys', norm=norm, alpha=0.5,
                                        edgecolors=cm.naviaS(2+1+j), linewidths=1)
                    ax[row, col + 3].scatter(mid_bins[is_measures],
                                                    np.diagonal(result['ihMTsat'][j])[is_measures],
                                                    c=np.diagonal(result['Nb_voxels'][j])[is_measures],
                                                    cmap='Greys', norm=norm,
                                                    edgecolors=cm.naviaS(2+1+j), linewidths=1)
                    ax[row, col + 3].scatter(mid_bins[is_not_measures],
                                        np.diagonal(result['ihMTsat'][j])[is_not_measures],
                                        c=np.diagonal(result['Nb_voxels'][j])[is_not_measures],
                                        cmap='Greys', norm=norm, alpha=0.5,
                                        edgecolors=cm.naviaS(2+1+j), linewidths=1)

                    # is_not_nan = np.diagonal(result['Nb_voxels'][j]) > 0
                    # weights = np.sqrt(np.diagonal(result['Nb_voxels'])[is_not_nan]) / np.max(np.diagonal(result['Nb_voxels'][j])[is_not_nan])
                    # mtr_fit = splrep(mid_bins[is_not_nan], np.diagonal(result['MTR'][j])[is_not_nan], w=weights, s=s_mtr[i])
                    # ax[row, col].plot(highres_bins, BSpline(*mtr_fit)(highres_bins), "--", color=cm.naviaS(2))
                    # mtsat_fit = splrep(mid_bins[is_not_nan], np.diagonal(result['MTsat'][j])[is_not_nan], w=weights, s=s_mtsat[i])
                    # ax[row, col + 1].plot(highres_bins, BSpline(*mtsat_fit)(highres_bins), "--", color=cm.naviaS(3))
                    # ihmtr_fit = splrep(mid_bins[is_not_nan], np.diagonal(result['ihMTR'][j])[is_not_nan], w=weights, s=s_ihmtr[i])
                    # ax[row, col + 2].plot(highres_bins, BSpline(*ihmtr_fit)(highres_bins), "--", color=cm.naviaS(4))
                    # ihmtsat_fit = splrep(mid_bins[is_not_nan], np.diagonal(result['ihMTsat'][j])[is_not_nan], w=weights, s=s_ihmtsat[i])
                    # ax[row, col + 3].plot(highres_bins, BSpline(*ihmtsat_fit)(highres_bins), "--", color=cm.naviaS(5))

                    # ax[row, col].set_ylim(0.975 * np.nanmin(np.diagonal(result['MTR'], axis1=1, axis2=2)),
                    #                     1.025 * np.nanmax(np.diagonal(result['MTR'], axis1=1, axis2=2)))
                    # ax[row, col].set_yticks([np.round(np.nanmin(np.diagonal(result['MTR'], axis1=1, axis2=2)), decimals=1),
                    #                         np.round(np.nanmax(np.diagonal(result['MTR'], axis1=1, axis2=2)), decimals=1)])
                    # ax[row, col].set_xlim(0, 90)
                    # # ax[row, col].tick_params(axis='y', labelcolor="C0")
                    # ax[row, col + 1].set_ylim(0.975 * np.nanmin(np.diagonal(result['MTsat'], axis1=1, axis2=2)),
                    #                         1.025 * np.nanmax(np.diagonal(result['MTsat'], axis1=1, axis2=2)))
                    # ax[row, col + 1].set_yticks([np.round(np.nanmin(np.diagonal(result['MTsat'], axis1=1, axis2=2)), decimals=1),
                    #                             np.round(np.nanmax(np.diagonal(result['MTsat'], axis1=1, axis2=2)), decimals=1)])
                    # ax[row, col + 1].set_xlim(0, 90)
                    # # ax[row, col + 1].tick_params(axis='y', labelcolor="C2")
                    # ax[row, col + 2].set_ylim(0.975 * np.nanmin(np.diagonal(result['ihMTR'], axis1=1, axis2=2)),
                    #                         1.025 * np.nanmax(np.diagonal(result['ihMTR'], axis1=1, axis2=2)))
                    # ax[row, col + 2].set_yticks([np.round(np.nanmin(np.diagonal(result['ihMTR'], axis1=1, axis2=2)), decimals=1),
                    #                             np.round(np.nanmax(np.diagonal(result['ihMTR'], axis1=1, axis2=2)), decimals=1)])
                    # ax[row, col + 2].set_xlim(0, 90)
                    # ax[row, col + 3].set_ylim(0.975 * np.nanmin(np.diagonal(result['ihMTsat'], axis1=1, axis2=2)),
                    #                         1.025 * np.nanmax(np.diagonal(result['ihMTsat'], axis1=1, axis2=2)))
                    # ax[row, col + 3].set_yticks([np.round(np.nanmin(np.diagonal(result['ihMTsat'], axis1=1, axis2=2)), decimals=1),
                    #                             np.round(np.nanmax(np.diagonal(result['ihMTsat'], axis1=1, axis2=2)), decimals=1)])
                    # ax[row, col + 3].set_xlim(0, 90)

                    bundle_idx += 1
        else:
            is_measures = whole_wm['Nb_voxels'] >= min_nb_voxels
            is_not_measures = np.invert(is_measures)
            ax[row, col].scatter(whole_mid_bins[is_measures],
                                whole_wm['MTR'][is_measures],
                                c=whole_wm['Nb_voxels'][is_measures],
                                cmap='Greys', norm=norm,
                                edgecolors=cm.naviaS(2), linewidths=1)
            ax[row, col].scatter(whole_mid_bins[is_not_measures],
                                 whole_wm['MTR'][is_not_measures],
                                 c=whole_wm['Nb_voxels'][is_not_measures],
                                 cmap='Greys', norm=norm, alpha=0.5,
                                 edgecolors=cm.naviaS(2), linewidths=1)
            ax[row, col + 1].scatter(whole_mid_bins[is_measures],
                                            whole_wm['MTsat'][is_measures],
                                            c=whole_wm['Nb_voxels'][is_measures],
                                            cmap='Greys', norm=norm,
                                            edgecolors=cm.naviaS(3), linewidths=1)
            ax[row, col + 1].scatter(whole_mid_bins[is_not_measures],
                                 whole_wm['MTsat'][is_not_measures],
                                 c=whole_wm['Nb_voxels'][is_not_measures],
                                 cmap='Greys', norm=norm, alpha=0.5,
                                 edgecolors=cm.naviaS(3), linewidths=1)
            ax[row, col + 2].scatter(whole_mid_bins[is_measures],
                                            whole_wm['ihMTR'][is_measures],
                                            c=whole_wm['Nb_voxels'][is_measures],
                                            cmap='Greys', norm=norm,
                                            edgecolors=cm.naviaS(4), linewidths=1)
            ax[row, col + 2].scatter(whole_mid_bins[is_not_measures],
                                 whole_wm['ihMTR'][is_not_measures],
                                 c=whole_wm['Nb_voxels'][is_not_measures],
                                 cmap='Greys', norm=norm, alpha=0.5,
                                 edgecolors=cm.naviaS(4), linewidths=1)
            ax[row, col + 3].scatter(whole_mid_bins[is_measures],
                                            whole_wm['ihMTsat'][is_measures],
                                            c=whole_wm['Nb_voxels'][is_measures],
                                            cmap='Greys', norm=norm,
                                            edgecolors=cm.naviaS(5), linewidths=1)
            ax[row, col + 3].scatter(whole_mid_bins[is_not_measures],
                                 whole_wm['ihMTsat'][is_not_measures],
                                 c=whole_wm['Nb_voxels'][is_not_measures],
                                 cmap='Greys', norm=norm, alpha=0.5,
                                 edgecolors=cm.naviaS(5), linewidths=1)

            is_not_nan = whole_wm['Nb_voxels'] > 0
            weights = np.sqrt(whole_wm['Nb_voxels'][is_not_nan]) / np.max(whole_wm['Nb_voxels'][is_not_nan])
            mtr_fit = splrep(whole_mid_bins[is_not_nan], whole_wm['MTR'][is_not_nan], w=weights, s=0.00001)
            ax[row, col].plot(highres_bins, BSpline(*mtr_fit)(highres_bins), "--", color=cm.naviaS(2))
            mtsat_fit = splrep(whole_mid_bins[is_not_nan], whole_wm['MTsat'][is_not_nan], w=weights, s=0.000001)
            ax[row, col + 1].plot(highres_bins, BSpline(*mtsat_fit)(highres_bins), "--", color=cm.naviaS(3))
            ihmtr_fit = splrep(whole_mid_bins[is_not_nan], whole_wm['ihMTR'][is_not_nan], w=weights, s=0.00005)
            ax[row, col + 2].plot(highres_bins, BSpline(*ihmtr_fit)(highres_bins), "--", color=cm.naviaS(4))
            ihmtsat_fit = splrep(whole_mid_bins[is_not_nan], whole_wm['ihMTsat'][is_not_nan], w=weights, s=0.0000005)
            ax[row, col + 3].plot(highres_bins, BSpline(*ihmtsat_fit)(highres_bins), "--", color=cm.naviaS(5))

            ax[row, col].set_ylim(0.975 * np.nanmin(whole_wm['MTR']),
                                  1.025 * np.nanmax(whole_wm['MTR']))
            ax[row, col].set_yticks([np.round(np.nanmin(whole_wm['MTR']), decimals=1),
                                     np.round(np.nanmax(whole_wm['MTR']), decimals=1)])
            ax[row, col].set_xlim(0, 90)
            # ax[row, col].tick_params(axis='y', labelcolor="C0")
            ax[row, col + 1].set_ylim(0.975 * np.nanmin(whole_wm['MTsat']),
                                      1.025 * np.nanmax(whole_wm['MTsat']))
            ax[row, col + 1].set_yticks([np.round(np.nanmin(whole_wm['MTsat']), decimals=1),
                                         np.round(np.nanmax(whole_wm['MTsat']), decimals=1)])
            ax[row, col + 1].set_xlim(0, 90)
            # ax[row, col + 1].tick_params(axis='y', labelcolor="C2")
            ax[row, col + 2].set_ylim(0.975 * np.nanmin(whole_wm['ihMTR']),
                                      1.025 * np.nanmax(whole_wm['ihMTR']))
            ax[row, col + 2].set_yticks([np.round(np.nanmin(whole_wm['ihMTR']), decimals=1),
                                         np.round(np.nanmax(whole_wm['ihMTR']), decimals=1)])
            ax[row, col + 2].set_xlim(0, 90)
            ax[row, col + 3].set_ylim(0.975 * np.nanmin(whole_wm['ihMTsat']),
                                      1.025 * np.nanmax(whole_wm['ihMTsat']))
            ax[row, col + 3].set_yticks([np.round(np.nanmin(whole_wm['ihMTsat']), decimals=1),
                                         np.round(np.nanmax(whole_wm['ihMTsat']), decimals=1)])
            ax[row, col + 3].set_xlim(0, 90)

        ax[row, col + 1].yaxis.set_label_position("right")
        ax[row, col + 1].yaxis.tick_right()
        ax[row, col + 3].yaxis.set_label_position("right")
        ax[row, col + 3].yaxis.tick_right()

        # PCC
        if args.MT_corr:
            ax[0, 0].text(1.015, 1.03, "PCC", color="dimgrey",
                          transform=ax[0, 0].transAxes, size=8)
            ax[row, 0].text(1.015, 0.45, f'{np.around(mt_corr[row], decimals=2):.2f}',
                            transform=ax[row, 0].transAxes, size=8, color="dimgrey")
        if args.ihMT_corr:
            ax[0, 2].text(1.015, 1.03, "PCC", color="dimgrey",
                          transform=ax[0, 2].transAxes, size=8)
            ax[row, 2].text(1.015, 0.45, f'{np.around(ihmt_corr[row], decimals=2):.2f}',
                            transform=ax[row, 2].transAxes, size=8, color="dimgrey")

        ax[row, col].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

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
            ax[row, col + 2].get_xaxis().set_ticks([])
            ax[row, col + 3].get_xaxis().set_ticks([])

        if row == 0:
            ax[row, col].title.set_text('MTR')
            ax[row, col + 1].title.set_text('MTsat')
            ax[row, col + 2].title.set_text('ihMTR')
            ax[row, col + 3].title.set_text('ihMTsat')

        ax[row, 0].set_ylabel(bundles_names[i], labelpad=10)
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

    line = plt.Line2D([0.477, 0.477], [0.035,0.985], transform=fig.transFigure, color="black", linestyle=(0, (5, 5)), alpha=0.7)
    fig.add_artist(line)
    # plt.show()
    plt.savefig(out_path1, dpi=500, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
