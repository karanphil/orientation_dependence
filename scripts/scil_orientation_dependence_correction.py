import argparse
import nibabel as nib
import numpy as np
from pathlib import Path

from modules.io import (extract_measures_as_list)
from modules.orientation_dependence import (correct_measure)

from scilpy.io.utils import (add_overwrite_arg)


# We suggest running scil_characterize_orientation_dependence.py again with
# the corrected data to make sure everything is in order.


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_peaks',
                   help='Path of the fODF peaks. The peaks are expected to be '
                        'given as unit directions.')
    p.add_argument('in_fixel_density_maps',
                   help='Path of the fixel density maps. This is the output '
                        'of the scil_bundle_fixel_analysis script, without '
                        'the split_bundles option. Thus, all the bundles '
                        'be present in the file, as a 5th dimension.')
    p.add_argument('out_folder',
                   help='Path of the output folder for txt, png, masks and '
                        'measures.')
    
    p.add_argument('--polyfits', nargs='+', required=True,
                   help='List of polyfit files. Should be the '
                        'output of the \n'
                        'scil_characterize_orientation_dependence.py script.')

    p.add_argument('--in_measures', nargs='+', required=True,
                   help='Path to the measures to correct.')
    p.add_argument('--measures_names', nargs='+', required=True,
                   help='Name of the measures to correct. Most match the '
                        'names used in the '
                        'scil_characterize_orientation_dependence.py script')
    
    p.add_argument('--lookuptable',
                   help='Path of the bundles lookup table, outputed by the '
                        'scil_fixel_density_maps script. Allows to make sure '
                        'the polyfits and fixel_density_maps follow the same '
                        'order.')    

    g = p.add_argument_group(title='Characterization parameters')
    g.add_argument('--min_frac_thr', default=0.1,
                   help='Value of the minimal fraction threshold for '
                        'selecting peaks to correct [%(default)s].')
    
    s1 = p.add_argument_group(title='Save differences')
    s1.add_argument('--save_differences', action='store_true',
                    help='If set, will save the difference between original '
                         'and corrected measures.')
    s1.add_argument('--differences_folder',
                    help='Output folder of where to save the differences.')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    out_folder = Path(args.out_folder)

    # Load the data
    peaks_img = nib.load(args.in_peaks)
    peaks = peaks_img.get_fdata()
    affine = peaks_img.affine

    fixel_density_maps_img = nib.load(args.in_fixel_density_maps)
    fixel_density_maps = fixel_density_maps_img.get_fdata()

    if args.lookuptable:
        lookuptable = np.loadtxt(args.lookuptable, dtype=str)[0]

    measures, measures_names = extract_measures_as_list(args.in_measures,
                                                        args.measures_names)
    
    for measure, measure_name in zip(measures, measures_names):
        polyfit_shape = np.load(args.polyfits[0])[measure_name + "_polyfit"].shape
        polyfits = np.ndarray((polyfit_shape) + (len(args.polyfits),))
        references = np.zeros(len(args.polyfits))
        bundles_names = np.empty(len(args.polyfits), dtype=object)
        for i, polyfit in enumerate(args.polyfits):
            bundle_name = Path(polyfit).parent.name
            if args.lookuptable:
                if bundle_name in lookuptable:
                    bundle_idx = np.argwhere(lookuptable == bundle_name)[0][0]
                else:
                    raise ValueError("Polyfit from bundle not present in lookup table.")
            else:
                bundle_idx = i
            polyfits[..., bundle_idx] = np.load(polyfit)[measure_name + "_polyfit"]
            references[bundle_idx] = np.load(polyfit)[measure_name + "_reference"]
            bundles_names[bundle_idx] = bundle_name

        if (lookuptable != bundles_names).all():
            raise ValueError("The order of polyfits and lookup table are not the same.")

        # Compute correction
        corrected_measure= correct_measure(measure, peaks, affine,
                                           polyfits, references,
                                           fixel_density_maps)

        # Save results
        corrected_path = out_folder / str(str(measure_name) + "_corrected.nii.gz")
        nib.save(nib.Nifti1Image(corrected_measure, affine), corrected_path)
        if args.save_differences:
            if args.differences_folder:
                diff_folder = Path(args.differences_folder)
            else:
                diff_folder = out_folder
            difference = corrected_measure - measure
            difference_path = diff_folder / str(str(measure_name) + "_difference.nii.gz")
            nib.save(nib.Nifti1Image(difference, affine), difference_path)


if __name__ == "__main__":
    main()
