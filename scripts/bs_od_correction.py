import argparse
import nibabel as nib
import numpy as np
from pathlib import Path

from modules.io import (extract_measures, plot_all_bundles_means)
from modules.orientation_dependence import (compute_single_fiber_means,
                                            fit_single_fiber_results)

from scilpy.io.utils import (add_overwrite_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_peaks',
                   help='Path of the fODF peaks. The peaks are expected to be '
                        'given as unit directions.')
    p.add_argument('in_peak_values',
                   help='Path of the fODF peak values. The peak values must '
                        'not be max-normalized for \neach voxel, but rather '
                        'they should keep the actual fODF amplitude of the '
                        'peaks.')
    p.add_argument('in_fa',
                   help='Path of the FA.')
    p.add_argument('in_nufo',
                   help='Path to the NuFO.')
    p.add_argument('in_wm_mask',
                   help='Path of the WM mask.')
    p.add_argument('out_folder',
                   help='Path of the output folder for txt, png, masks and '
                        'measures.')
    
    p.add_argument('--measures', nargs='+', default=[],
                   action='append', required=True,
                   help='List of measures to characterize.')
    p.add_argument('--measures_names', nargs='+', default=[], action='append',
                   help='List of names for the measures to characterize.')
    p.add_argument('--bundles', nargs='+', default=[],
                   action='append', required=True,
                   help='Path to the bundle ROIs for where to analyze.')
    p.add_argument('--bundles_names', nargs='+', default=[], action='append',
                   help='List of names of the bundles.')
    
    p.add_argument('--in_e1',
                   help='Path to the principal eigenvector of DTI.')

    g = p.add_argument_group(title='Characterization parameters')
    g.add_argument('--fa_thr', default=0.5,
                   help='Value of FA threshold [%(default)s].')
    g.add_argument('--bin_width_1f', default=5, type=int,
                   help='Value of the bin width for the single-fiber '
                        'characterization [%(default)s].')
    g.add_argument('--min_frac_thr', default=0.1,
                   help='Value of the minimal fraction threshold for '
                        'selecting peaks to correct [%(default)s].')
    g.add_argument('--min_nb_voxels', default=1, type=int,
                   help='Value of the minimal number of voxels per bin '
                        '[%(default)s].')
    g.add_argument('--poly_order', default=8, type=int,
                   help='Order of the polynome to fit [%(default)s].')
    
    s1 = p.add_argument_group(title='Save npz files')
    s1.add_argument('--save_npz_files', action='store_true',
                    help='If set, will save the results as npz files.')
    s1.add_argument('--npz_folder',
                    help='Output folder of where to save the npz files.')
    
    s2 = p.add_argument_group(title='Save plots')
    s2.add_argument('--save_plots', action='store_true',
                    help='If set, will save the results as plots.')
    s2.add_argument('--plots_folder',
                    help='Output folder of where to save the plots.')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    nb_measures = len(args.measures[0])
    if args.measures_names != [] and\
        (len(args.measures_names[0]) != nb_measures):
        parser.error('When using --measures_names, you need to specify ' +
                     'the same number of measures as given in --measures.')

    out_folder = Path(args.out_folder)

    # Load the data
    peaks_img = nib.load(args.in_peaks)
    peak_values_img = nib.load(args.in_peak_values)
    fa_img = nib.load(args.in_fa)
    nufo_img = nib.load(args.in_nufo)
    wm_mask_img = nib.load(args.in_wm_mask)

    peaks = peaks_img.get_fdata()
    peak_values = peak_values_img.get_fdata()
    fa = fa_img.get_fdata()
    nufo = nufo_img.get_fdata()
    wm_mask = wm_mask_img.get_fdata()

    affine = peaks_img.affine

    min_nb_voxels = args.min_nb_voxels

    if args.in_e1:
        e1_img = nib.load(args.in_e1)
        e1 = e1_img.get_fdata()
    else:
        e1 = peaks

    measures, measures_names = extract_measures(args.measures,
                                                fa.shape,
                                                args.measures_names)
    
    bundles, bundles_names = extract_measures(args.bundles,
                                              fa.shape,
                                              args.bundles_names)

    #----------------------- Single-fiber section -----------------------------
    print("Computing single-fiber means.")
    polyfits = np.zeros((args.poly_order + 1, nb_measures,
                         len(bundles_names)))
    max_measures = np.zeros((nb_measures, len(bundles_names)))
    is_measures = np.ones((len(np.arange(0, 90 + args.bin_width_1f,
                                         args.bin_width_1f)) - 1,
                            len(bundles_names)), dtype=bool)
    means = np.zeros((len(np.arange(0, 90 + args.bin_width_1f,
                                    args.bin_width_1f)) - 1,
                      nb_measures, len(bundles_names)))
    voxel_counts = np.zeros((len(np.arange(0, 90 + args.bin_width_1f,
                                           args.bin_width_1f)) - 1,
                             len(bundles_names)))
    max_count = 0
    for i in range(bundles.shape[-1]):
        print("Processing {} bundle.".format(bundles_names[i]))
        bins, means[..., i], voxel_counts[..., i] =\
            compute_single_fiber_means(e1, fa,
                                       wm_mask,
                                       affine,
                                       measures,
                                       nufo=nufo,
                                       mask=bundles[..., i],
                                       bin_width=args.bin_width_1f,
                                       fa_thr=args.fa_thr)

        curr_max_count = np.max(voxel_counts[..., i])
        if curr_max_count > max_count:
            max_count = curr_max_count

        is_measures[..., i] = voxel_counts[..., i] >= min_nb_voxels
        nb_bins = np.sum(is_measures[..., i])
        if nb_bins == 0:
            msg = """No angle bin was filled above the required minimum number
                of voxels. The script was unable to produce a single-fiber
                characterization of the measures. The {} bundle probably
                contains too few single-fiber voxels. Try to carefully reduce
                the min_nb_voxels.""".format(bundles_names[i])
            raise ValueError(msg)
        min_bin = bins[:-1][is_measures[..., i]][0]
        max_bin = bins[1:][is_measures[..., i]][-1]
        bin_range = np.arange(min_bin, max_bin + 1, 1)

        weights = np.sqrt(voxel_counts[..., i])
    
        print("Fitting the whole brain results.")
        polyfits[..., i] = fit_single_fiber_results(bins,
                                                    means[..., i],
                                                    poly_order=args.poly_order,
                                                    is_measures=is_measures[..., i],
                                                    weights=weights,
                                                    scale_poly_order=True)

        for j in range(nb_measures):
            polynome = np.poly1d(polyfits[:, j, i])
            max_measures[j, i] = np.max(polynome(bin_range))

    if args.save_plots:
        if args.plots_folder:
            plots_folder = Path(args.plots_folder)
        else:
            plots_folder = out_folder
        print("Saving single-fiber results as plots.")
        bundles_order = np.concatenate((bundles_names[np.argwhere(bundles_names != 'MCP')].squeeze(),
                                        np.array(['MCP'])))
        for i in range(nb_measures):
            plot_all_bundles_means(bins, means[:, i, :], voxel_counts,
                                is_measures, max_count, polyfits[:, i, :],
                                bundles_names, measures_names[i], plots_folder,
                                bundles_order=bundles_order)

    # TODO : adapt these for multi-bundle.
    # print("Saving polyfit results.")
    # for i in range(nb_measures):
    #     out_path = out_folder / str(str(measures_name[i]) + "_polyfit.npy")
    #     np.save(out_path, measures_fit[:, i])
    
    # if args.save_npz_files:
    #     if args.npz_folder:
    #         npz_folder = Path(args.npz_folder)
    #     else:
    #         npz_folder = out_folder
    #     out_path = npz_folder / '1f_original_results'
    #     print("Saving results as npz files.")
    #     save_results_as_npz(bins, measure_means, nb_voxels,
    #                         measures_name, out_path)


if __name__ == "__main__":
    main()
