import argparse
import nibabel as nib
import numpy as np
from pathlib import Path

from modules.io import (extract_measures)
from modules.orientation_dependence import (correct_measure)

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
    
    p.add_argument('--delta_m_max_fit', nargs='+', default=[], action='append',
                   help='Path of the delta_m_max fit files. Should be the '
                        'output of the \n'
                        'scil_characterize_orientation_dependence.py script.')

    g = p.add_argument_group(title='Characterization parameters')
    g.add_argument('--min_frac_thr', default=0.1,
                   help='Value of the minimal fraction threshold for '
                        'selecting peaks to correct [%(default)s].')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.measures_names != [] and\
        (len(args.measures_names[0]) != len(args.measures[0])):
        parser.error('When using --measures_names, you need to specify ' +
                     'the same number of measures as given in --measures.')

    out_folder = Path(args.out_folder)

    # Load the data
    peaks_img = nib.load(args.in_peaks)
    peak_values_img = nib.load(args.in_peak_values)
    wm_mask_img = nib.load(args.in_wm_mask)

    peaks = peaks_img.get_fdata()
    peak_values = peak_values_img.get_fdata()
    wm_mask = wm_mask_img.get_fdata()

    affine = peaks_img.affine

    measures, measures_name = extract_measures(args.measures, wm_mask.shape,
                                               args.measures_names)

    for i in range(measures.shape[-1]):
        corrected_measure = correct_measure(peaks, peak_values,
                                            measures[..., i], affine,
                                            wm_mask, sf_polyfit,
                                            peak_frac_thr=args.min_frac_thr,
                                            delta_m_max_fit=delta_m_max_fit)
    corrected_path = out_folder / str(str(measures_name[i]) + "_corrected.nii.gz")
    nib.save(nib.Nifti1Image(corrected_measure, affine), corrected_path)

if __name__ == "__main__":
    main()
