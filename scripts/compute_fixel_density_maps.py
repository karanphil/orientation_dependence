import argparse
import nibabel as nib
import numpy as np
from pathlib import Path

from dipy.io.stateful_tractogram import StatefulTractogram

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

    p.add_argument('--in_bundles', nargs='+', default=[],
                   action='append', required=True,
                   help='Path to the bundle trk for where to analyze.')
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

    max_theta = 45
    thrs_value = 0.00

    nb_bundles = len(args.in_bundles[0])

    fixel_density_maps = np.zeros((peaks.shape[:-1]) + (5, nb_bundles))
    fixel_density_masks = np.zeros((peaks.shape[:-1]) + (5, nb_bundles))
    
    for i, bundle in enumerate(args.in_bundles[0]):
        print(bundle)
        sft = load_tractogram_with_reference(parser, args, bundle)

        sft.to_vox()
        sft.to_corner()

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

            normalized_seg = np.reshape(segments / seg_lengths[..., None], (-1, 3))

            for vox_idx, seg_dir in zip(vox_indices, normalized_seg):
                vox_idx = tuple(vox_idx)
                peaks_at_idx = peaks[vox_idx].reshape((5,3))

                cos_theta = np.abs(np.dot(seg_dir.reshape((-1, 3)),
                                        peaks_at_idx.T))

                if (cos_theta > min_cos_theta).any():
                    lobe_idx = np.argmax(np.squeeze(cos_theta), axis=0)  # (n_segs)
                    fixel_density_maps[vox_idx][lobe_idx][i] += 1

        maps_thrs = thrs_value * np.max(fixel_density_maps[..., i])
        fixel_density_masks[..., i] = np.where(fixel_density_maps[..., i] > maps_thrs,
                                               1, 0)

    nb_bundles_per_fixel = np.sum(fixel_density_masks, axis=-1)
    nb_unique_bundles_per_fixel = np.where(np.sum(fixel_density_masks,
                                                  axis=-2) > 0, 1, 0)
    nb_bundles_per_voxel = np.sum(nb_unique_bundles_per_fixel, axis=-1)

    nib.save(nib.Nifti1Image(fixel_density_maps.astype(np.uint8),
             affine), out_folder / "fixel_density_maps.nii.gz")
    
    nib.save(nib.Nifti1Image(fixel_density_masks.astype(np.uint8),
             affine), out_folder / "fixel_density_masks.nii.gz")
    
    nib.save(nib.Nifti1Image(nb_bundles_per_fixel.astype(np.uint8),
             affine), out_folder / "nb_bundles_per_fixel.nii.gz")
    
    nib.save(nib.Nifti1Image(nb_bundles_per_voxel.astype(np.uint8),
             affine), out_folder / "nb_bundles_per_voxel.nii.gz")

if __name__ == "__main__":
    main()
