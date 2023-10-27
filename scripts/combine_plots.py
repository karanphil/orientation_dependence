import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from modules.io import plot_init

from scilpy.io.utils import (add_overwrite_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('out_folder',
                   help='Path of the output folder for txt, png, masks and '
                        'measures.')
    
    p.add_argument('--results', nargs='+', default=[],
                   action='append', required=True,
                   help='List of characterization results.')
    p.add_argument('--polyfits', nargs='+', default=[],
                   action='append', required=True,
                   help='List of polyfits.')
    p.add_argument('--bundles_names', nargs='+', default=[], action='append',
                   help='List of names for the characterized bundles.')

    g = p.add_argument_group(title='Characterization parameters')
    g.add_argument('--min_nb_voxels', default=30, type=int,
                   help='Value of the minimal number of voxels per bin '
                        '[%(default)s].')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    nb_results = len(args.results[0])
    if len(args.polyfits[0]) != nb_results:
        parser.error('When using --polyfits, you need to specify ' +
                     'the same number of polyfits as given in --results.')

    out_folder = Path(args.out_folder)

    results = []
    extracted_bundles = []
    polyfits = []
    max_count = 0
    for i, result in enumerate(args.results[0]):
        print("Loading: ", result)
        results.append(np.load(result))
        extracted_bundles.append(str(Path(result).parent))
        polyfits.append(np.load(args.polyfits[0][i]))
        curr_max_count = np.max(results[i]['Nb_voxels'])
        if curr_max_count > max_count:
            max_count = curr_max_count
    
    measure = Path(args.polyfits[0][i]).name.split("_")[0]
    if args.bundles_names != []:
        bundles_names = args.bundles_names[0]
    else:
        bundles_names = extracted_bundles

    if "MCP" in bundles_names:
        bundles_names.remove("MCP")
        bundles_names.append("MCP")
    nb_bundles = len(bundles_names)
    nb_rows = int(np.ceil(nb_bundles / 2))

    mid_bins = (results[0]['Angle_min'] + results[0]['Angle_max']) / 2.
    highres_bins = np.arange(0, 90 + 1, 0.5)

    out_path = out_folder / str("all_bundles_original_" + str(measure) + "_1f.png")
    plot_init(dims=(8, 10), font_size=10)
    fig, ax = plt.subplots(nb_rows, 2, layout='constrained')
    bundle_idx = 0
    for i in range(nb_bundles):
        col = i % 2
        row = i // 2
        if bundles_names[i] in extracted_bundles:
            result = results[bundle_idx]
            norm = mpl.colors.Normalize(vmin=0, vmax=max_count)
            # WARNING!!! I must adapt this script to the new way of saving the results (with all bins not None).
            colorbar = ax[row, col].scatter(mid_bins, result[measure],
                                            c=result['Nb_voxels'],
                                            cmap='Greys', norm=norm,
                                            edgecolors="C0", linewidths=1)
            # if cr_means is not None:
            #     ax1.scatter(mid_bins, cr_means[..., i], c=nb_voxels, cmap='Greys',
            #                 norm=norm, edgecolors="C0", linewidths=1, marker="s")
            #     out_path = out_folder / str("corrected_" + str(names[i]) + "_1f.png")
            polynome = np.poly1d(polyfits[bundle_idx])
            ax[row, col].plot(highres_bins, polynome(highres_bins), "--", color="C0")
            ax[row, col].set_ylim(0.975 * np.nanmin(result[measure]),
                                  1.025 * np.nanmax(result[measure]))
            ax[row, col].set_yticks([np.round(np.nanmin(result[measure]), decimals=1),
                                     np.round(np.nanmax(result[measure]), decimals=1)])
            ax[row, col].set_xlim(0, 90)
            bundle_idx += 1
        else:
            ax[row, col].set_yticks([])
        ax[row, col].legend(handles=[colorbar], labels=[bundles_names[i]],
                            loc='center left', bbox_to_anchor=(1, 0.5),
                            markerscale=0, handletextpad=-2.0, handlelength=2)
        if row != nb_rows - 1:
            ax[row, col].get_xaxis().set_ticks([])
        if row == (nb_rows - 1) / 2:
            ax[row, 0].set_ylabel(str(measure) + ' mean')
            ax[row, 0].yaxis.set_label_coords(-0.25, 0.5)
    fig.colorbar(colorbar, ax=ax[:, 1], location='right',
                 label="Voxel count", aspect=100)
    ax[nb_rows - 1, 0].set_xlabel(r'$\theta_a$')
    ax[nb_rows - 1, 0].set_xlim(0, 90)
    ax[nb_rows - 1, 0].set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax[nb_rows - 1, 1].set_xlabel(r'$\theta_a$')
    ax[nb_rows - 1, 1].set_xlim(0, 90)
    ax[nb_rows - 1, 1].set_xticks([0, 15, 30, 45, 60, 75, 90])
    if nb_bundles % 2 != 0:
        ax[nb_rows - 1, 1].set_yticks([])
    fig.get_layout_engine().set(h_pad=0, hspace=0)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
