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
    p.add_argument('out_folder',
                   help='Path of the output folder for txt, png, masks and '
                        'measures.')
    
    p.add_argument('--results', nargs='+', default=[],
                   action='append', required=True,
                   help='List of characterization results.')
    p.add_argument('--bundle_names', nargs='+', default=[], action='append',
                   help='List of names for the characterized bundles.')
    p.add_argument('--polyfits', nargs='+', default=[],
                   action='append',
                   help='List of polyfits.')

    g = p.add_argument_group(title='Characterization parameters')
    g.add_argument('--min_nb_voxels', default=30, type=int,
                   help='Value of the minimal number of voxels per bin '
                        '[%(default)s].')

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


if __name__ == "__main__":
    main()
