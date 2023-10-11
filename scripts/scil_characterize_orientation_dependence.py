import argparse
import nibabel as nib
import numpy as np
from pathlib import Path

from modules.orientation_dependence import (compute_single_fiber_averages)

from scilpy.io.utils import (add_overwrite_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_peaks',
                   help='Path of the fODF peaks.')
    p.add_argument('in_peak_values',
                   help='Path of the fODF peak values.')
    p.add_argument('in_fa',
                   help='Path of the FA.')
    p.add_argument('in_wm_mask',
                   help='Path of the WM mask.')
    p.add_argument('in_nufo',
                   help='Path to the NuFO.')
    p.add_argument('out_folder',
                   help='Path of the output folder for txt, png, masks and measures.')
    
    p.add_argument('--in_e1',
                   help='Path to the principal eigenvector of DTI.')
    p.add_argument('--in_v1',
                   help='Path to the principal eigenvalue of DTI.')
    p.add_argument('--in_mtr',
                   help='Path to the MTR.')
    p.add_argument('--in_ihmtr',
                   help='Path to the ihMTR.')
    p.add_argument('--in_mtsat',
                   help='Path to the MTsat.')
    p.add_argument('--in_ihmtsat',
                   help='Path to the ihMTsat.')
    p.add_argument('--in_roi',
                   help='Path to the ROI for single fiber analysis.')

    p.add_argument('--files_basename',
                   help='Basename of all the saved txt or png files.')

    p.add_argument('--fa_thr', default=0.5,
                   help='Value of FA threshold [%(default)s].')
    p.add_argument('--bin_width', default=1,
                   help='Value of the bin width for the whole brain [%(default)s].')
    p.add_argument('--bin_width_mask', default=3,
                   help='Value of the bin width inside the mask [%(default)s].')
    p.add_argument('--bin_width_bundles', default=5,
                   help='Value of the bin width inside bundles [%(default)s].')
    # p.add_argument('--frac_thr', default=0.4,
    #                help='Value of the fraction threshold for selecting 2 fibers [%(default)s].')
    p.add_argument('--min_frac_thr', default=0.1,
                   help='Value of the minimal fraction threshold for selecting peaks to correct [%(default)s].')
    p.add_argument('--min_nb_voxels', default=30, type=int,
                   help='Value of the minimal number of voxels per bin [%(default)s].')
    p.add_argument('--poly_order', default=10,
                   help='Order of the polynome to fit [%(default)s].')
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.files_basename:
        files_basename = args.files_basename
    else:
        files_basename = "_" + str(args.fa_thr) + "_fa_thr_" \
            + str(args.bin_width) + "_bin_width"
        
    out_folder = Path(args.out_folder)

    # Load the data
    peaks_img = nib.load(args.in_peaks)
    peak_values_img = nib.load(args.in_peak_values)
    fa_img = nib.load(args.in_fa)
    wm_mask_img = nib.load(args.in_wm_mask)
    nufo_img = nib.load(args.in_nufo)


    peaks = peaks_img.get_fdata()
    peak_values = peak_values_img.get_fdata()
    fa = fa_img.get_fdata()
    wm_mask = wm_mask_img.get_fdata()
    nufo = nufo_img.get_fdata()

    affine = peaks_img.affine

    if args.in_e1:
        e1_img = nib.load(args.in_e1)
        e1 = e1_img.get_fdata()
    else:
        e1 = peaks

    if args.compute_angle_maps:
        print("Computing angle maps.")
        save_angle_map(e1, fa, wm_mask, affine, out_folder, peaks, peak_values, nufo)

    print("Computing single fiber averages.")
    w_brain_results = compute_single_fiber_averages(e1, fa,
                                                    wm_mask,
                                                    affine,
                                                    mtr=mtr,
                                                    ihmtr=ihmtr,
                                                    mtsat=mtsat,
                                                    ihmtsat=ihmtsat,
                                                    nufo=nufo,
                                                    bin_width=args.bin_width,
                                                    fa_thr=args.fa_thr,
                                                    min_nb_voxels=args.min_nb_voxels)