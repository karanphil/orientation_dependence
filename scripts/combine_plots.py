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
    p.add_argument('--bundles_names', nargs='+', default=[], action='append',
                   required=True,
                   help='List of names for the characterized bundles.')
    p.add_argument('--polyfits', nargs='+', default=[],
                   action='append', required=True,
                   help='List of polyfits.')
    p.add_argument('--measures_names', nargs='+', default=[], action='append',
                   required=True,
                   help='List of names the characterized measures.')

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
    if args.bundles_names != [] and (len(args.bundles_names[0]) != nb_results):
        parser.error('When using --bundles_names, you need to specify ' +
                     'the same number of bundles as given in --results.')

    if args.polyfits != [] and (len(args.polyfits[0]) != nb_results):
        parser.error('When using --polyfits, you need to specify ' +
                     'the same number of polyfits as given in --results.')

    out_folder = Path(args.out_folder)

    results = []
    bundles_name = []
    polyfits = []
    for i, result in enumerate(args.results[0]):
        results.append(np.load(result))
        bundles_name.append(args.bundles_names[0][i])
        polyfits.append(np.load(result)) # Il y a plus qu'un polyfit file par bundle?
    
    nb_bundles = len(results)
    nb_measures = len(results[0].files()) - 3
    measures_names = args.measures_names[0]

    mid_bins = (results[0]['Angle_min'] + results[0]['Angle_max']) / 2.
    highres_bins = np.arange(0, 90 + 1, 0.5)
    plot_init()
    for measure in measures_names:
        out_path = out_folder / str("all_bundles_original_" + str(measure) + "_1f.png")
        fig, (ax, cax) = plt.subplots(nb_bundles, 2,
                                        gridspec_kw={"width_ratios":[1, 0.05]})
        for j, result in enumerate(results):
            max_count = np.max(result['Nb_voxels'])
            norm = mpl.colors.Normalize(vmin=0, vmax=max_count)
            colorbar = ax[j].scatter(mid_bins, result[measure], c=result['Nb_voxels'],
                                     cmap='Greys', norm=norm,
                                     edgecolors="C0", linewidths=1)
            # if cr_means is not None:
            #     ax1.scatter(mid_bins, cr_means[..., i], c=nb_voxels, cmap='Greys',
            #                 norm=norm, edgecolors="C0", linewidths=1, marker="s")
            #     out_path = out_folder / str("corrected_" + str(names[i]) + "_1f.png")
            polynome = np.poly1d(polyfits[j])
            ax[j].plot(highres_bins, polynome(highres_bins), "--", color="C0")
            ax[j].set_xlabel(r'$\theta_a$')
            ax[j].set_xlim(0, 90)
            ax[j].set_ylim(0.975 * np.nanmin(result[measure]), 1.025 * np.nanmax(result[measure]))
            ax[j].set_ylabel(str(measure) + ' mean')
            ax[j].colorbar(colorbar, cax=cax[j], label="Voxel count")
        fig.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
