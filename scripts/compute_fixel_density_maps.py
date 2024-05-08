import argparse
import nibabel as nib
import numpy as np
from pathlib import Path

from dipy.io.streamline import load_tractogram
from scilpy.io.utils import (add_overwrite_arg, add_processes_arg,
                             add_reference_arg)
from scilpy.tractanalysis.grid_intersections import grid_intersections

from modules.fixel_analysis import compute_fixel_density


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
    
    p.add_argument('--maps_thr', default=0.0,
                   help='Value of density maps threshold [%(default)s].')
    
    p.add_argument('--norm', default="fixel", choices=["fixel", "voxel"],
                   help='Way of normalizing the density maps [%(default)s].')

    add_overwrite_arg(p)
    add_processes_arg(p)
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
    thrs_value = args.maps_thr
    
    fixel_density_maps = compute_fixel_density(peaks, max_theta,
                                               args.in_bundles[0],
                                               nbr_processes=args.nbr_processes)

    # fixel_density_masks = np.zeros(fixel_density_maps.shape)

    # Currently, this applies a threshold on the number of streamlines.
    # maps_thrs = thrs_value * np.max(fixel_density_maps[..., i])
    fixel_density_masks = fixel_density_maps > thrs_value

    # Normalizing the density maps
    voxel_sum = np.sum(np.sum(fixel_density_maps, axis=-1), axis=-1)
    fixel_sum = np.sum(fixel_density_maps, axis=-1)

    for i, bundle in enumerate(args.in_bundles[0]):
        bundle_name = Path(bundle).name.split(".")[0]

        if args.norm == "voxel":
            fixel_density_maps[..., 0, i] /= voxel_sum
            fixel_density_maps[..., 1, i] /= voxel_sum
            fixel_density_maps[..., 2, i] /= voxel_sum
            fixel_density_maps[..., 3, i] /= voxel_sum
            fixel_density_maps[..., 4, i] /= voxel_sum
        
        elif args.norm == "fixel":
            fixel_density_maps[..., i] /= fixel_sum

        nib.save(nib.Nifti1Image(fixel_density_maps[..., i],
                                 affine),
                 out_folder / "fixel_density_maps_{}.nii.gz".format(bundle_name))

    nb_bundles_per_fixel = np.sum(fixel_density_masks, axis=-1)
    nb_unique_bundles_per_fixel = np.where(np.sum(fixel_density_masks,
                                                  axis=-2) > 0, 1, 0)
    nb_bundles_per_voxel = np.sum(nb_unique_bundles_per_fixel, axis=-1)

    nib.save(nib.Nifti1Image(fixel_density_maps,
             affine), out_folder / "fixel_density_maps.nii.gz")
    
    nib.save(nib.Nifti1Image(fixel_density_masks.astype(np.uint8),
             affine), out_folder / "fixel_density_masks.nii.gz")
    
    nib.save(nib.Nifti1Image(nb_bundles_per_fixel.astype(np.uint8),
             affine), out_folder / "nb_bundles_per_fixel.nii.gz")
    
    nib.save(nib.Nifti1Image(nb_bundles_per_voxel.astype(np.uint8),
             affine), out_folder / "nb_bundles_per_voxel.nii.gz")

if __name__ == "__main__":
    main()
