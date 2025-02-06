import argparse
import nibabel as nib
import numpy as np
import logging
import pandas as pd
from pathlib import Path

from scilpy.io.utils import add_verbose_arg

from modules.io import (save_profiles_as_npz, extract_measures)
from modules.utils import compute_sf_mf_mask


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('out_folder',
                   help='Path of the output folder for txt, png, masks and '
                        'measures.')

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--in_fixel_density_masks',
                   help='Path of the fixel density masks. This is the output '
                        'of the scil_bundle_fixel_analysis script, without '
                        'the split_bundles option. Thus, all the bundles '
                        'be present in the file, as a 5th dimension.')
    g.add_argument('--in_nufo',
                   help='Path of the NuFO map.')

    p.add_argument('--measures', nargs='+', default=[], required=True,
                   help='List of measures to characterize.')
    p.add_argument('--measures_names', nargs='+', default=[],
                   help='List of names for the measures to characterize.')

    p.add_argument('--bundles', nargs='+', default=[], required=True,
                   help='Path to the bundles labels for where to analyze.')
    p.add_argument('--bundles_names', nargs='+', default=[],
                   help='List of names for the bundles.')

    p.add_argument('--suffix', default='', type=str,
                   help='Suffix to add at the end of the saved file')

    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    nb_measures = len(args.measures)
    if args.measures_names != [] and\
        (len(args.measures_names) != nb_measures):
        parser.error('When using --measures_names, you need to specify ' +
                     'the same number of measures as given in --measures.')

    out_folder = Path(args.out_folder)

    # Load the data
    if args.in_fixel_density_masks:
        fixel_density_masks_img = nib.load(args.in_fixel_density_masks)
        fixel_density_masks = fixel_density_masks_img.get_fdata()
        sf_mask, mf_mask = compute_sf_mf_mask(fixel_density_masks)
    elif args.in_nufo:
        nufo_img = nib.load(args.in_nufo)
        nufo = nufo_img.get_fdata()
        mf_mask = nufo > 1
        sf_mask = nufo == 1

    measures, measures_name = extract_measures(args.measures,
                                               fixel_density_masks.shape[0:3],
                                               args.measures_names)

    bundles = []
    bundles_names = []
    for bundle in args.bundles:
        bundles.append(nib.load(bundle).get_fdata())
        bundles_names.append(Path(bundle).name.split(".")[0])

    if args.bundles_names:
        bundles_names = args.bundles_names

    nb_bundles = len(bundles_names)
    nb_measures = measures.shape[-1]

    nb_sections = int(np.max(bundles[0]))
    measure_means = np.zeros((nb_bundles, nb_sections, nb_measures, 3))
    measure_stds = np.zeros((nb_bundles, nb_sections, nb_measures, 3))
    nb_voxels = np.zeros((nb_bundles, nb_sections, nb_measures, 3))

    # For every section of every bundle, compute the mean measures
    for i, (bundle, bundle_name) in enumerate(zip(bundles, bundles_names)):
        logging.info("Computing multi-fiber means of bundle {}.".format(bundle_name))
        for j in range(nb_sections):
            section_mask = (bundle == (j + 1)) & (sf_mask > 0 | mf_mask > 0)
            sf_section_mask = (sf_mask > 0) & section_mask
            mf_section_mask = (mf_mask > 0) & section_mask
            measure_means[i, j, :, 0] = np.nanmean(measures[section_mask], axis=0)
            measure_stds[i, j, :, 0] = np.nanstd(measures[section_mask], axis=0)
            nb_voxels[i, j, :, 0] = np.nansum(section_mask)
            measure_means[i, j, :, 1] = np.nanmean(measures[sf_section_mask], axis=0)
            measure_stds[i, j, :, 1] = np.nanstd(measures[sf_section_mask], axis=0)
            nb_voxels[i, j, :, 1] = np.nansum(sf_section_mask)
            measure_means[i, j, :, 2] = np.nanmean(measures[mf_section_mask], axis=0)
            measure_stds[i, j, :, 2] = np.nanstd(measures[mf_section_mask], axis=0)
            nb_voxels[i, j, :, 2] = np.nansum(mf_section_mask)

    # Saving the results of orientation dependence characterization
    for i in range(nb_bundles):
        out_path = out_folder / (bundles_names[i] + '/tract_profiles_' + args.suffix)
        save_profiles_as_npz(measure_means[i], measure_stds[i],
                             nb_voxels[i], measures_name, out_path)



if __name__ == "__main__":
    main()
