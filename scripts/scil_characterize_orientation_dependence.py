import argparse
import nibabel as nib
import numpy as np
from pathlib import Path

from modules.io import (plot_means, save_angle_maps, save_masks_by_angle_bins,
                        save_results_as_txt)
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
                   help='Path of the output folder for txt, png, masks and measures.')
    
    p.add_argument('--measures', nargs='+', default=[],
                   action='append', required=True,
                   help='List of measures to characterize.')
    
    p.add_argument('--in_e1',
                   help='Path to the principal eigenvector of DTI.')
    p.add_argument('--in_roi',
                   help='Path to the ROI for single fiber analysis.')

    g = p.add_argument_group(title='Optional parameters')
    g.add_argument('--fa_thr', default=0.5,
                   help='Value of FA threshold [%(default)s].')
    g.add_argument('--bin_width', default=1,
                   help='Value of the bin width for the whole brain [%(default)s].')
    # p.add_argument('--frac_thr', default=0.4,
    #                help='Value of the fraction threshold for selecting 2 fibers [%(default)s].')
    g.add_argument('--min_frac_thr', default=0.1,
                   help='Value of the minimal fraction threshold for selecting peaks to correct [%(default)s].')
    g.add_argument('--min_nb_voxels', default=30, type=int,
                   help='Value of the minimal number of voxels per bin [%(default)s].')
    g.add_argument('--poly_order', default=10,
                   help='Order of the polynome to fit [%(default)s].')
    
    p1 = p.add_argument_group(title='Save angle info')
    p1.add_argument('--save_angle_info', action='store_true',
                    help='If set, will save the angle maps and masks.')
    p1.add_argument('--angle_folder',
                    help='Output folder of where to save the angle info.')
    
    p2 = p.add_argument_group(title='Save txt files')
    p2.add_argument('--save_txt_files', action='store_true',
                    help='If set, will save the results as txt files.')
    p2.add_argument('--txt_folder',
                    help='Output folder of where to save the txt files.')
    
    p3 = p.add_argument_group(title='Save plots')
    p3.add_argument('--save_plots', action='store_true',
                    help='If set, will save the results as plots.')
    p3.add_argument('--plots_folder',
                    help='Output folder of where to save the plots.')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.files_basename:
        files_basename = args.files_basename
    else:
        files_basename = "_"
        
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

    measures = np.ndarray((fa.shape) + (len(args.measures),))
    measures_name = np.ndarray((len(args.measures),), dtype=object)
    for i, measure in enumerate(args.measures):
        measures[..., i] = (nib.load(measure)).get_fdata()
        measures_name[i] = Path(measure).name.split(".")[0]

    print("Computing single-fiber means.")
    bins, measure_means, nb_voxels =\
        compute_single_fiber_means(e1, fa,
                                   wm_mask,
                                   affine,
                                   measures,
                                   nufo=nufo,
                                   mask=roi,
                                   bin_width=args.bin_width,
                                   fa_thr=args.fa_thr,
                                   min_nb_voxels=args.min_nb_voxels)
    
    print("Fitting the whole brain results.")
    measures_fit = fit_single_fiber_results(bins, measure_means,
                                             poly_order=args.poly_order)
    print("Saving polyfit results.")
    for i in range(measures_fit.shape[-1]):
        out_path = out_folder / str(str(measures_name[i]) + "_polyfit.npy")
        np.save(out_path, measures_fit[:, i])
    
    if args.save_txt_files:
        if args.txt_folder:
            txt_folder = Path(args.txt_folder)
        else:
            txt_folder = out_folder
        print("Saving results as txt files.")
        save_results_as_txt(bins, measure_means, nb_voxels, measures_name,
                            txt_folder)
        
    if args.save_plots:
        if args.plots_folder:
            plots_folder = Path(args.plots_folder)
        else:
            plots_folder = out_folder
        print("Saving single-fiber results as plots.")
        plot_means(bins, measures, nb_voxels, measures_name,
                   plots_folder, polyfit=measures_fit)
        
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
                                 nufo=nufo, bin_width=10, fa_thr=args.fa_thr)
    
if __name__ == "__main__":
    main()
