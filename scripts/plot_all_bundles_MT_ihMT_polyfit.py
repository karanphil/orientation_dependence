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
    
    p.add_argument('--results', nargs='+', default=[],
                   action='append', required=True,
                   help='List of characterization results.')
    p.add_argument('--bundles_names', nargs='+', default=[], action='append',
                   required=True,
                   help='List of names for the characterized bundles.')

    p.add_argument('--whole_WM', default=[],
                   help='Path to the whole WM characterization.')
    
    p.add_argument('--out_name', default='toto.png',
                   help='Name of the output file.')

    p.add_argument('--measures', default='MT')

    p.add_argument('--corr')  

    g = p.add_argument_group(title='Characterization parameters')
    g.add_argument('--min_nb_voxels', default=30, type=int,
                   help='Value of the minimal number of voxels per bin '
                        '[%(default)s].')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    nb_results = len(args.results[0])

    out_folder = Path(args.out_folder)
    min_nb_voxels = args.min_nb_voxels

    bundles_names = args.bundles_names[0]

    results = []
    extracted_bundles = []
    max_count = 0
    tmp = 0
    for i, result in enumerate(args.results[0]):
        if str(Path(result).parent) in bundles_names:
            print("Loading: ", result)
            results.append(np.load(result))
            extracted_bundles.append(str(Path(result).parent))
            curr_max_count = np.max(results[tmp]['Nb_voxels'])
            if curr_max_count > max_count:
                max_count = curr_max_count
            tmp += 1

    if args.whole_WM:
        print("Loading: ", args.whole_WM)
        whole_wm = np.load(args.whole_WM)
        whole_mid_bins = (whole_wm['Angle_min'] + whole_wm['Angle_max']) / 2.

    if "ICP_L" in bundles_names:
        bundles_names.remove("ICP_L")
    if "ICP_R" in bundles_names:
        bundles_names.remove("ICP_R")

    if "MCP" in bundles_names:
        bundles_names.remove("MCP")
        bundles_names.append("MCP")

    bundles_names.append("WM")

    mt_corr = None
    if args.corr:
        mt_corr = np.loadtxt(args.corr)

    nb_bundles = len(bundles_names)
    nb_rows = int(np.ceil(nb_bundles / 2))
    # nb_rows = nb_bundles

    s_r = np.ones(nb_bundles) * 0.0005
    s_sat = np.ones(nb_bundles) * 0.0005
    if args.measures == 'MT':
        measures = ['MTR', 'MTsat']
        cmap_idx = [2, 3]

        s_r[3] = 0.00001 # CC_2a
        s_r[5] = 0.001 # CC_3
        s_r[11] = 0.0015 # CG_R
        s_r[12] = 0.0002 # CR_L
        s_r[13] = 0.00005 # CR_R
        s_r[14] = 0.0002 # CST_L
        s_r[15] = 0.00005 # CST_R
        s_r[18] = 0.0001 # ILF_L
        s_r[19] = 0.00003 # IFL_R
        s_r[22] = 0.01 # SLF_1_L
        s_r[24] = 0.005 # SLF_2_L
        s_r[26] = 0.005 # SLF_3_L
        s_r[27] = 0.005 # SLF_3_R
        # s_r[16] = 0.01 # ICP_L
        # s_r[17] = 0.05 # ICP_R
        # s_r[20] = 0.0001 # ILF_L
        # s_r[21] = 0.00003 # IFL_R
        # s_r[24] = 0.01 # SLF_1_L
        # s_r[26] = 0.005 # SLF_2_L
        # s_r[28] = 0.005 # SLF_3_L
        # s_r[29] = 0.005 # SLF_3_R

        s_sat[0] = 0.00001 # AF_L
        s_sat[1] = 0.0001 # AF_R
        s_sat[5] = 0.0001 # CC_3
        s_sat[6] = 0.00005 # CC_4
        s_sat[11] = 0.0001 # CG_R
        s_sat[12] = 0.00001 # CR_L
        s_sat[14] = 0.00001 # CST_L
        s_sat[16] = 0.00001 # IFOF_L
        s_sat[18] = 0.00001 # ILF_L
        s_sat[21] = 0.00005 # OR_R
        s_sat[24] = 0.0001 # SLF_2_L
        s_sat[28] = 0.0001 # UF_L
        s_sat[30] = 0.0003 # MCP
        # s_sat[16] = 0.005 # ICP_L
        # s_sat[17] = 0.05 # ICP_R
        # s_sat[18] = 0.00001 # IFOF_L
        # s_sat[20] = 0.00001 # ILF_L
        # s_sat[23] = 0.00005 # OR_R
        # s_sat[26] = 0.0001 # SLF_2_L
        # s_sat[30] = 0.0001 # UF_L
        # s_sat[32] = 0.0003 # MCP

    if args.measures == 'ihMT':
        measures = ['ihMTR', 'ihMTsat']
        cmap_idx = [4, 5]

        s_r[5] = 0.001 # CC_3
        s_r[11] = 0.003 # CG_R
        s_r[18] = 0.00005 # ILF_L
        s_r[20] = 0.00005 # OR_L
        s_r[22] = 0.05 # SLF_1_L
        s_r[26] = 0.005 # SLF_3_L
        s_r[28] = 0.005 # UF_L
        s_r[29] = 0.005 # UF_R
        # s_r[16] = 0.005 # ICP_L
        # s_r[17] = 0.05 # ICP_R
        # s_r[20] = 0.00005 # ILF_L
        # s_r[22] = 0.00005 # OR_L
        # s_r[24] = 0.05 # SLF_1_L
        # s_r[28] = 0.005 # SLF_3_L
        # s_r[30] = 0.005 # UF_L
        # s_r[31] = 0.005 # UF_R

        s_sat[2] = 0.00001 # CC_1
        s_sat[6] = 0.00001 # CC_4
        s_sat[7] = 0.000005 # CC_5
        s_sat[8] = 0.00001 # CC_6
        s_sat[9] = 0.0000005 # CC_7
        s_sat[11] = 0.00005 # CG_R
        s_sat[13] = 0.000005 # CR_R
        s_sat[15] = 0.000003 # CST_R
        s_sat[18] = 0.000001 # ILF_L
        s_sat[20] = 0.000001 # OR_L
        s_sat[23] = 0.00001 # SLF_1_R
        s_sat[25] = 0.00001 # SLF_2_R
        s_sat[27] = 0.00005 # SLF_3_R
        s_sat[30] = 0.00001 # MCP
        # s_sat[17] = 0.001 # ICP_R
        # s_sat[20] = 0.000001 # ILF_L
        # s_sat[22] = 0.000001 # OR_L
        # s_sat[25] = 0.00001 # SLF_1_R
        # s_sat[27] = 0.00001 # SLF_2_R
        # s_sat[29] = 0.00005 # SLF_3_R
        # s_sat[32] = 0.00001 # MCP

    mid_bins = (results[0]['Angle_min'] + results[0]['Angle_max']) / 2.
    highres_bins = np.arange(0, 90 + 1, 0.5)

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
        col = i % 2
        if col == 1:
            col = 2
        row = i // 2
        if bundles_names[i] in extracted_bundles:
            bundle_idx = extracted_bundles.index(bundles_names[i])
            result = results[bundle_idx]
            is_measures = result['Nb_voxels'] >= min_nb_voxels
            is_not_measures = np.invert(is_measures)
            norm = mpl.colors.Normalize(vmin=0, vmax=max_count)
            colorbar = ax[row, col].scatter(mid_bins[is_measures],
                                            result[measures[0]][is_measures],
                                            c=result['Nb_voxels'][is_measures],
                                            cmap='Greys', norm=norm,
                                            edgecolors=cm.naviaS(cmap_idx[0]), linewidths=1)
            ax[row, col].scatter(mid_bins[is_not_measures],
                                 result[measures[0]][is_not_measures],
                                 c=result['Nb_voxels'][is_not_measures],
                                 cmap='Greys', norm=norm, alpha=0.5,
                                 edgecolors=cm.naviaS(cmap_idx[0]), linewidths=1)
            ax[row, col + 1].scatter(mid_bins[is_measures],
                                            result[measures[1]][is_measures],
                                            c=result['Nb_voxels'][is_measures],
                                            cmap='Greys', norm=norm,
                                            edgecolors=cm.naviaS(cmap_idx[1]), linewidths=1)
            ax[row, col + 1].scatter(mid_bins[is_not_measures],
                                 result[measures[1]][is_not_measures],
                                 c=result['Nb_voxels'][is_not_measures],
                                 cmap='Greys', norm=norm, alpha=0.5,
                                 edgecolors=cm.naviaS(cmap_idx[1]), linewidths=1)

            is_not_nan = result['Nb_voxels'] > 0
            weights = np.sqrt(result['Nb_voxels'][is_not_nan]) / np.max(result['Nb_voxels'][is_not_nan])
            mtr_fit = splrep(mid_bins[is_not_nan], result[measures[0]][is_not_nan], w=weights, s=s_r[i])
            ax[row, col].plot(highres_bins, BSpline(*mtr_fit)(highres_bins), "--", color=cm.naviaS(cmap_idx[0]))
            mtsat_fit = splrep(mid_bins[is_not_nan], result[measures[1]][is_not_nan], w=weights, s=s_sat[i])
            ax[row, col + 1].plot(highres_bins, BSpline(*mtsat_fit)(highres_bins), "--", color=cm.naviaS(cmap_idx[1]))

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
            ax[row, col].scatter(whole_mid_bins[is_measures],
                                whole_wm[measures[0]][is_measures],
                                c=whole_wm['Nb_voxels'][is_measures],
                                cmap='Greys', norm=norm,
                                edgecolors=cm.naviaS(cmap_idx[0]), linewidths=1)
            ax[row, col].scatter(whole_mid_bins[is_not_measures],
                                 whole_wm[measures[0]][is_not_measures],
                                 c=whole_wm['Nb_voxels'][is_not_measures],
                                 cmap='Greys', norm=norm, alpha=0.5,
                                 edgecolors=cm.naviaS(cmap_idx[0]), linewidths=1)
            ax[row, col + 1].scatter(whole_mid_bins[is_measures],
                                            whole_wm[measures[1]][is_measures],
                                            c=whole_wm['Nb_voxels'][is_measures],
                                            cmap='Greys', norm=norm,
                                            edgecolors=cm.naviaS(cmap_idx[1]), linewidths=1)
            ax[row, col + 1].scatter(whole_mid_bins[is_not_measures],
                                 whole_wm[measures[1]][is_not_measures],
                                 c=whole_wm['Nb_voxels'][is_not_measures],
                                 cmap='Greys', norm=norm, alpha=0.5,
                                 edgecolors=cm.naviaS(cmap_idx[1]), linewidths=1)

            is_not_nan = whole_wm['Nb_voxels'] > 0
            weights = np.sqrt(whole_wm['Nb_voxels'][is_not_nan]) / np.max(whole_wm['Nb_voxels'][is_not_nan])
            mtr_fit = splrep(whole_mid_bins[is_not_nan], whole_wm[measures[0]][is_not_nan], w=weights, s=0.00001)
            ax[row, col].plot(highres_bins, BSpline(*mtr_fit)(highres_bins), "--", color=cm.naviaS(cmap_idx[0]))
            mtsat_fit = splrep(whole_mid_bins[is_not_nan], whole_wm[measures[1]][is_not_nan], w=weights, s=0.000001)
            ax[row, col + 1].plot(highres_bins, BSpline(*mtsat_fit)(highres_bins), "--", color=cm.naviaS(cmap_idx[1]))

            ax[row, col].set_ylim(0.975 * np.nanmin(whole_wm[measures[0]]),
                                  1.025 * np.nanmax(whole_wm[measures[0]]))
            ax[row, col].set_yticks([np.round(np.nanmin(whole_wm[measures[0]]), decimals=1),
                                     np.round(np.nanmax(whole_wm[measures[0]]), decimals=1)])
            ax[row, col].set_xlim(0, 90)
            # ax[row, col].tick_params(axis='y', labelcolor="C0")
            ax[row, col + 1].set_ylim(0.975 * np.nanmin(whole_wm[measures[1]]),
                                      1.025 * np.nanmax(whole_wm[measures[1]]))
            ax[row, col + 1].set_yticks([np.round(np.nanmin(whole_wm[measures[1]]), decimals=1),
                                         np.round(np.nanmax(whole_wm[measures[1]]), decimals=1)])
            ax[row, col + 1].set_xlim(0, 90)

        ax[row, col + 1].yaxis.set_label_position("right")
        ax[row, col + 1].yaxis.tick_right()

        # PCC
        if args.corr:
            ax[0, col].text(1.015, 1.03, "PCC", color="dimgrey",
                          transform=ax[0, col].transAxes, size=8)
            ax[row, col].text(1.015, 0.45, f'{np.around(mt_corr[i], decimals=2):.2f}',
                            transform=ax[row, col].transAxes, size=8, color="dimgrey")

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

        if row == 0:
            ax[row, col].title.set_text(measures[0])
            ax[row, col + 1].title.set_text(measures[1])

        if bundles_names[i].split('_')[0] == 'SLF':
            fontsize = 7
        else:
            fontsize = 9
        ax[row, col].set_ylabel(bundles_names[i], labelpad=10, fontsize=fontsize)
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
    plt.savefig(out_path1, dpi=500, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
