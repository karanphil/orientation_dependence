import argparse
import nibabel as nib
import numpy as np
from pathlib import Path
import logging

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
    
    p.add_argument('--abs_thr', default=None, type=int,
                   help='Value of density maps threshold to obtain density '
                        'masks, in number of streamlines [%(default)s].')
    
    p.add_argument('--rel_thr', default=None, type=float,
                   help='Value of density maps threshold to obtain density '
                        'masks, as a ratio of the normalized density '
                        '[%(default)s].')
    
    p.add_argument('--max_theta', default=45,
                   help='Maximum angle between streamline and peak to be '
                        'associated [%(default)s].')
    
    p.add_argument('--save_masks', action='store_true',
                   help='If set, save the density masks for each bundle.')
    
    p.add_argument('--select_single_bundle', action='store_true',
                   help='If set, select the voxels where only one bundle is '
                        'present.')
    
    p.add_argument('--norm', default="fixel", choices=["fixel", "voxel"],
                   help='Way of normalizing the density maps [%(default)s].')

    add_overwrite_arg(p)
    add_processes_arg(p)
    add_reference_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.abs_thr is None and args.rel_thr is None:
        logging.error("Need one of abs_thr and rel_thr.")

    out_folder = Path(args.out_folder)

    # Load the data
    peaks_img = nib.load(args.in_peaks)

    peaks = peaks_img.get_fdata()

    affine = peaks_img.affine

    bundles = []
    for bundle in args.in_bundles[0]:
        if "CR" in bundle:
            print("Removing bundle ", bundle)
        else:
            bundles.append(bundle)
    
    fixel_density_maps = compute_fixel_density(peaks, args.max_theta, bundles,
                                               nbr_processes=args.nbr_processes)

    # This applies a threshold on the number of streamlines.
    fixel_density_masks_abs = np.ones(fixel_density_maps.shape)
    if args.abs_thr is not None:
        fixel_density_masks_abs = fixel_density_maps > args.abs_thr

    # Normalizing the density maps
    voxel_sum = np.sum(np.sum(fixel_density_maps, axis=-1), axis=-1)
    fixel_sum = np.sum(fixel_density_maps, axis=-1)

    for i, bundle in enumerate(bundles):
        bundle_name = Path(bundle).name.split(".")[0]

        if args.norm == "voxel":
            fixel_density_maps[..., 0, i] /= voxel_sum
            fixel_density_maps[..., 1, i] /= voxel_sum
            fixel_density_maps[..., 2, i] /= voxel_sum
            fixel_density_maps[..., 3, i] /= voxel_sum
            fixel_density_maps[..., 4, i] /= voxel_sum
        
        elif args.norm == "fixel":
            fixel_density_maps[..., i] /= fixel_sum

        nib.save(nib.Nifti1Image(fixel_density_maps[..., i], affine),
                 out_folder / "fixel_density_maps_{}.nii.gz".format(bundle_name))
    
    # This applies a threshold on the normalized density (percentage).
    fixel_density_masks_rel = np.ones(fixel_density_maps.shape)
    if args.rel_thr is not None:
        fixel_density_masks_rel = fixel_density_maps > args.rel_thr

    fixel_density_masks = fixel_density_masks_rel * fixel_density_masks_abs

    # Compute number of bundles per fixel
    nb_bundles_per_fixel = np.sum(fixel_density_masks, axis=-1)
    # Compute a mask of the present of each bundle
    nb_unique_bundles_per_fixel = np.where(np.sum(fixel_density_masks,
                                                  axis=-2) > 0, 1, 0)
    # Compute number of bundles per fixel by taking the sum of the mask
    nb_bundles_per_voxel = np.sum(nb_unique_bundles_per_fixel, axis=-1)
    # Single-fiber single-bundle voxels
    single_bundle_per_voxel = nb_bundles_per_voxel == 1

    for i, bundle in enumerate(bundles):
        bundle_name = Path(bundle).name.split(".")[0]

        if args.save_masks:
            nib.save(nib.Nifti1Image(fixel_density_masks[..., i].astype(np.uint8), affine),
                     out_folder / "fixel_density_masks_{}.nii.gz".format(bundle_name))
            if args.select_single_bundle:
                bundle_mask = fixel_density_masks[..., 0, i] * single_bundle_per_voxel
                nib.save(nib.Nifti1Image(bundle_mask.astype(np.uint8), affine),
                         out_folder / "bundle_mask_only_{}.nii.gz".format(bundle_name))

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
