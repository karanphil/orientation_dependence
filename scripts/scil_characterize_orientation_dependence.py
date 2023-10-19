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
    p.add_argument('--in_roi',
                   help='Path to the ROI for where to analyze.')

    p.add_argument('--compute_three_fiber_crossings', action='store_true',
                   help='If set, will perform the three-fiber crossings '
                        'analysis.')

    g = p.add_argument_group(title='Characterization parameters')
    g.add_argument('--fa_thr', default=0.5,
                   help='Value of FA threshold [%(default)s].')
    g.add_argument('--bin_width_1f', default=1, type=int,
                   help='Value of the bin width for the single-fiber '
                        'characterization [%(default)s].')
    g.add_argument('--bin_width_2f', default=10, type=int,
                   help='Value of the bin width for the two-fiber '
                        'characterization [%(default)s].')
    g.add_argument('--bin_width_3f', default=30, type=int,
                   help='Value of the bin width for the three-fiber '
                        'characterization [%(default)s].')
    g.add_argument('--min_frac_thr', default=0.1,
                   help='Value of the minimal fraction threshold for '
                        'selecting peaks to correct [%(default)s].')
    g.add_argument('--min_nb_voxels', default=30, type=int,
                   help='Value of the minimal number of voxels per bin '
                        '[%(default)s].')
    g.add_argument('--poly_order', default=10, type=int,
                   help='Order of the polynome to fit [%(default)s].')
    
    s1 = p.add_argument_group(title='Save angle info')
    s1.add_argument('--save_angle_info', action='store_true',
                    help='If set, will save the angle maps and masks.')
    s1.add_argument('--angle_folder',
                    help='Output folder of where to save the angle info.')
    s1.add_argument('--angle_mask_bin_width', default=10, type=int,
                    help='Bin width used for the angle masks [%(default)s].')
    
    s2 = p.add_argument_group(title='Save npz files')
    s2.add_argument('--save_npz_files', action='store_true',
                    help='If set, will save the results as npz files.')
    s2.add_argument('--npz_folder',
                    help='Output folder of where to save the npz files.')
    
    s3 = p.add_argument_group(title='Save plots')
    s3.add_argument('--save_plots', action='store_true',
                    help='If set, will save the results as plots.')
    s3.add_argument('--plots_folder',
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

    if args.in_e1:
        e1_img = nib.load(args.in_e1)
        e1 = e1_img.get_fdata()
    else:
        e1 = peaks

    if args.in_roi:
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
    bins, measure_means, nb_voxels =\
        compute_single_fiber_means(e1, fa,
                                   wm_mask,
                                   affine,
                                   measures,
                                   nufo=nufo,
                                   mask=roi,
                                   bin_width=args.bin_width_1f,
                                   fa_thr=args.fa_thr,
                                   min_nb_voxels=args.min_nb_voxels)

    not_nan = np.isfinite(measure_means)
    if np.sum(not_nan) == 0:
        msg = """No angle bin was filled above the required minimum number of
              voxels. The script was unable to produce a single-fiber
              characterization of the measures. If --in_roi was used, the
              region of interest probably contains too few single-fiber
              voxels. Try to carefully reduce the min_nb_voxels."""
        raise ValueError(msg)
    
    print("Fitting the whole brain results.")
    measures_fit = fit_single_fiber_results(bins,
                                            measure_means[:, :nb_measures],
                                            poly_order=args.poly_order)
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
                   plots_folder, polyfit=measures_fit)
        if correction:
            plot_means(bins, measure_means[:, :nb_measures], nb_voxels,
                       measures_name, plots_folder, polyfit=measures_fit,
                       cr_means=measure_means[:, nb_measures:])
        
    if args.save_angle_info:
        if args.angle_folder:
            angle_folder = Path(args.angle_folder)
        else:
            angle_folder = out_folder
        print("Saving angle maps.")
        save_angle_maps(e1, fa, wm_mask, affine, angle_folder,
                        peaks, peak_values, nufo)
        print("Saving single-fiber masks.")
        save_masks_by_angle_bins(e1, fa, wm_mask, affine, angle_folder,
                                 nufo=nufo, fa_thr=args.fa_thr,
                                 bin_width=args.angle_mask_bin_width)
        
    # Compute single-fiber delta_m_max
    sf_delta_m_max = np.nanmax(measure_means[:, :nb_measures], axis=0) -\
        np.nanmin(measure_means[:, :nb_measures], axis=0)

    #---------------------- Crossing fibers section ---------------------------
    print("Computing two-fiber means.")
    bins, measure_means, nb_voxels, labels =\
        compute_two_fibers_means(peaks, peak_values,
                                        wm_mask, affine,
                                        nufo, measures, mask=roi,
                                        bin_width=args.bin_width_2f,
                                        min_nb_voxels=args.min_nb_voxels)
    
    measure_means_diag = np.diagonal(measure_means, axis1=1, axis2=2)
    measure_means_diag = np.swapaxes(measure_means_diag, 1, 2)
    nb_voxels_diag = np.diagonal(nb_voxels, axis1=1, axis2=2)

    print("Analysing two-fiber delta_m_max.")
    slope, origin, delta_m_max, frac_thrs_mid, min_bins, max_bins =\
        analyse_delta_m_max(bins, measure_means_diag[..., :nb_measures],
                            sf_delta_m_max, nb_voxels_diag)
    print("Analysis found these bins as minima and maxima: ")
    for i in range(nb_measures):
        print(str(measures_name[i]) + " minimum at " + str(min_bins[i]) + " degrees")
        print(str(measures_name[i]) + " maximum at " + str(max_bins[i]) + " degrees")

    print("Saving delta_m_max fit results.")
    for i in range(slope.shape[-1]):
        out_path = out_folder / str(str(measures_name[i]) + "_delta_m_max_fit.npy")
        np.save(out_path, np.concatenate(([origin[i]], [slope[i]])))

    if args.save_npz_files:
        print("Saving results as npz files.")
        out_path = npz_folder / "2f_original_results"
        save_results_as_npz(bins, measure_means[..., :nb_measures], nb_voxels,
                            measures_name, out_path)
        if correction:
            out_path = npz_folder / '2f_corrected_results'
            save_results_as_npz(bins, measure_means[..., nb_measures:],
                                nb_voxels, measures_name, out_path)

    if args.save_plots:
        print("Saving two-fiber results as plots.")
        plot_3d_means(bins, measure_means[0, :, :, :nb_measures], plots_folder,
                      measures_name, nametype="original")
        plot_multiple_means(bins, measure_means_diag[..., :nb_measures],
                            nb_voxels_diag, plots_folder, measures_name,
                            labels=labels, legend_title=r"Peak$_1$ fraction",
                            endname="2D_2f", delta_max=delta_m_max,
                            delta_max_slope=slope, delta_max_origin=origin,
                            p_frac=frac_thrs_mid, nametype="original")
        if correction:
            plot_3d_means(bins, measure_means[0, :, :, nb_measures:],
                          plots_folder, measures_name, nametype="corrected")
            plot_multiple_means(bins, measure_means_diag[..., nb_measures:],
                                nb_voxels_diag, plots_folder, measures_name,
                                labels=labels,
                                legend_title=r"Peak$_1$ fraction",
                                endname="2D_2f", markers='s',
                                nametype="corrected")

    if args.compute_three_fiber_crossings:
        print("Computing 3 crossing fibers means.")
        bins, measure_means, nb_voxels, labels =\
            compute_three_fibers_means(peaks, peak_values, wm_mask, affine,
                                    nufo, measures, bin_width=args.bin_width_3f,
                                    min_nb_voxels=args.min_nb_voxels, mask=roi)
        
        if args.save_npz_files:
            print("Saving results as npz files.")
            out_path = npz_folder / "3f_original_results"
            save_results_as_npz(bins, measure_means[..., :nb_measures],
                                nb_voxels, measures_name, out_path)
            if correction:
                out_path = npz_folder / '3f_corrected_results'
                save_results_as_npz(bins, measure_means[..., nb_measures:],
                                    nb_voxels, measures_name, out_path)

        if args.save_plots:
            print("Saving three-fiber results as plots.")
            plot_multiple_means(bins, measure_means[..., :nb_measures],
                                nb_voxels, plots_folder,
                                measures_name, endname="2D_3f", labels=labels,
                                legend_title=r"Peak$_1$ fraction",
                                nametype="original")
            if correction:
                plot_multiple_means(bins, measure_means[..., nb_measures:],
                                    nb_voxels, plots_folder,
                                    measures_name, endname="2D_3f",
                                    labels=labels,
                                    legend_title=r"Peak$_1$ fraction",
                                    nametype="corrected", markers='s')

if __name__ == "__main__":
    main()
