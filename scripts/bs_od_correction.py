import argparse
import nibabel as nib
import numpy as np
from pathlib import Path

from modules.io import (plot_means, plot_3d_means, plot_multiple_means,
                        save_angle_maps, save_masks_by_angle_bins,
                        save_results_as_npz, extract_measures)
from modules.orientation_dependence import (analyse_delta_m_max,
                                            compute_three_fibers_means,
                                            compute_two_fibers_means,
                                            compute_single_fiber_means,
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
    p.add_argument('--measures_corrected', nargs='+', default=[],
                   action='append',
                   help='List of corrected measures to characterize.')
    
    p.add_argument('--in_e1',
                   help='Path to the principal eigenvector of DTI.')
    p.add_argument('--in_roi', nargs='+', default=[], action='append',
                   help='Path to the ROI or bundles for where to analyze.')

    g = p.add_argument_group(title='Characterization parameters')
    g.add_argument('--fa_thr', default=0.5,
                   help='Value of FA threshold [%(default)s].')
    g.add_argument('--bin_width_1f', default=1, type=int,
                   help='Value of the bin width for the single-fiber '
                        'characterization [%(default)s].')
    g.add_argument('--min_frac_thr', default=0.1,
                   help='Value of the minimal fraction threshold for '
                        'selecting peaks to correct [%(default)s].')
    g.add_argument('--min_nb_voxels', default=30, type=int,
                   help='Value of the minimal number of voxels per bin '
                        '[%(default)s].')
    g.add_argument('--use_weighted_polyfit', action='store_true',
                   help='If set, use weights when performing the polyfit. '
                        '[%(default)s].')
    g.add_argument('--poly_order', default=15, type=int,
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

    if args.measures_corrected != [] and\
        (len(args.measures_corrected[0]) != nb_measures):
        parser.error('When using --measures_corrected, you need to specify ' +
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

    if args.in_roi: # change that with extract_measures function.
        roi_img = nib.load(args.in_roi)
        roi = roi_img.get_fdata()
    else:
        roi = None

    measures_original, measures_name = extract_measures(args.measures,
                                                        fa.shape,
                                                        args.measures_names)

    if args.measures_corrected != []:
        correction = True
        measures_corrected, _ = extract_measures(args.measures_corrected,
                                                 fa.shape)
        measures = np.ndarray((fa.shape) + (nb_measures * 2,))
        measures[..., :nb_measures] = measures_original
        measures[..., nb_measures:] = measures_corrected
    else:
        correction = False
        measures = measures_original

    #----------------------- Single-fiber section -----------------------------
    print("Computing single-fiber means.")
    # Loop over all rois and save everything in a big np array.
    bins, measure_means, nb_voxels =\
        compute_single_fiber_means(e1, fa,
                                   wm_mask,
                                   affine,
                                   measures,
                                   nufo=nufo,
                                   mask=roi,
                                   bin_width=args.bin_width_1f,
                                   fa_thr=args.fa_thr)

    is_measures = nb_voxels >= min_nb_voxels
    nb_bins = np.sum(is_measures)
    if nb_bins == 0:
        msg = """No angle bin was filled above the required minimum number of
              voxels. The script was unable to produce a single-fiber
              characterization of the measures. If --in_roi was used, the
              region of interest probably contains too few single-fiber
              voxels. Try to carefully reduce the min_nb_voxels."""
        raise ValueError(msg)
    
    if args.use_weighted_polyfit:
        weights = np.sqrt(nb_voxels)
    else:
        weights = None
    
    print("Fitting the whole brain results.")
    measures_fit = fit_single_fiber_results(bins,
                                            measure_means[:, :nb_measures],
                                            poly_order=args.poly_order,
                                            is_measures=is_measures,
                                            weights=weights)

    print("Saving polyfit results.")
    for i in range(nb_measures):
        out_path = out_folder / str(str(measures_name[i]) + "_polyfit.npy")
        np.save(out_path, measures_fit[:, i])
    
    if args.save_npz_files:
        if args.npz_folder:
            npz_folder = Path(args.npz_folder)
        else:
            npz_folder = out_folder
        out_path = npz_folder / '1f_original_results'
        print("Saving results as npz files.")
        save_results_as_npz(bins, measure_means[:, :nb_measures], nb_voxels,
                            measures_name, out_path)
        if correction:
            out_path = npz_folder / '1f_corrected_results'
            save_results_as_npz(bins, measure_means[:, nb_measures:], nb_voxels,
                                measures_name, out_path)
    if args.save_plots:
        if args.plots_folder:
            plots_folder = Path(args.plots_folder)
        else:
            plots_folder = out_folder
        print("Saving single-fiber results as plots.")
        plot_means(bins, measure_means[:, :nb_measures], nb_voxels, measures_name,
                   plots_folder, polyfit=measures_fit, is_measures=is_measures)
        if correction:
            plot_means(bins, measure_means[:, :nb_measures], nb_voxels,
                       measures_name, plots_folder, polyfit=measures_fit,
                       cr_means=measure_means[:, nb_measures:],
                       is_measures=is_measures)


if __name__ == "__main__":
    main()
