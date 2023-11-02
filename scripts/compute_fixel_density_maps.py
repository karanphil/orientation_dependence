import argparse
import nibabel as nib
import numpy as np
from pathlib import Path

from dipy.io.stateful_tractogram import StatefulTractogram

from modules.io import (extract_measures, plot_all_bundles_means)
from modules.orientation_dependence import (compute_single_fiber_means,
                                            fit_single_fiber_results)

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg)
from scilpy.tractanalysis.grid_intersections import grid_intersections


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_peaks',
                   help='Path of the fODF peaks. The peaks are expected to be '
                        'given as unit directions.')
    p.add_argument('out_folder',
                   help='Path of the output folder for txt, png, masks and '
                        'measures.')

    p.add_argument('--bundles', nargs='+', default=[],
                   action='append', required=True,
                   help='Path to the bundle ROIs for where to analyze.')
    p.add_argument('--bundles_names', nargs='+', default=[], action='append',
                   help='List of names of the bundles.')

    add_overwrite_arg(p)
    add_reference_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    out_folder = Path(args.out_folder)

    # Load the data
    peaks_img = nib.load(args.in_peaks)

    peaks = peaks_img.get_fdata()

    affine = peaks_img.affine
    
    bundles, bundles_names = extract_measures(args.bundles,
                                              peaks.shape[:-1],
                                              args.bundles_names)
    
    nb_bundles = bundles_names.shape[0]

    max_theta = 60
    
    for i in range(nb_bundles):
        sft = load_tractogram_with_reference(parser, args, bundles[i])

        sft.to_vox()
        sft.to_corner()

        metric_sum_map = np.zeros(metric.shape[:-1])
        weight_map = np.zeros(metric.shape[:-1])
        min_cos_theta = np.cos(np.radians(max_theta))

        all_crossed_indices = grid_intersections(sft.streamlines)
        for crossed_indices in all_crossed_indices:
            segments = crossed_indices[1:] - crossed_indices[:-1]
            seg_lengths = np.linalg.norm(segments, axis=1)

            # Remove points where the segment is zero.
            # This removes numpy warnings of division by zero.
            non_zero_lengths = np.nonzero(seg_lengths)[0]
            segments = segments[non_zero_lengths]
            seg_lengths = seg_lengths[non_zero_lengths]

            # Those starting points are used for the segment vox_idx computations
            seg_start = crossed_indices[non_zero_lengths]
            vox_indices = (seg_start + (0.5 * segments)).astype(int)

            normalization_weights = np.ones_like(seg_lengths)

            normalized_seg = np.reshape(segments / seg_lengths[..., None], (-1, 3))

            for vox_idx, seg_dir, norm_weight in zip(vox_indices,
                                                    normalized_seg,
                                                    normalization_weights):
                vox_idx = tuple(vox_idx)
                peaks_at_idx = peaks[vox_idx]

                cos_theta = np.abs(np.dot(seg_dir.reshape((-1, 3)),
                                        peaks_at_idx.T))

                metric_val = 0.0
                if (cos_theta > min_cos_theta).any():
                    lobe_idx = np.argmax(np.squeeze(cos_theta), axis=0)  # (n_segs)
                    metric_val = metric[vox_idx][lobe_idx]

                metric_sum_map[vox_idx] += metric_val * norm_weight
                weight_map[vox_idx] += norm_weight


if __name__ == "__main__":
    main()
