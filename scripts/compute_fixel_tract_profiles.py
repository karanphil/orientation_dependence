import argparse
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
import logging

from scilpy.io.utils import (add_overwrite_arg, add_processes_arg,
                             add_reference_arg, add_verbose_arg)

from modules.io import (extract_measures)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_csv', help='Path of the input csv.')
    p.add_argument('out_csv', help='Path of the output csv.')
    
    p.add_argument('--measures', nargs='+', default=[], required=True,
                   help='List of measures to characterize.')
    p.add_argument('--measures_names', nargs='+', default=[],
                   help='List of names for the measures to characterize.')

    p.add_argument('--in_bundles_labels', nargs='+', default=[], required=True,
                   help='')
    p.add_argument('--bundles_names', nargs='+', default=[],
                   help='List of names for the bundles.')

    p.add_argument('--in_weights', required=True,
                   help='')

    p.add_argument('--type', type=str)

    add_overwrite_arg(p)
    add_processes_arg(p)
    add_reference_arg(p)
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

    df = pd.read_csv(args.in_csv)

    df = df[df['Sid'] == 'sub-026-hc_ses-3']

    data_shape = (nib.load(args.measures[0])).get_fdata().shape

    measures, measures_name = extract_measures(args.measures,
                                               data_shape,
                                               args.measures_names)

    measures[measures == 0] = None

    bundles_labels = []
    bundles_names = []
    for label in args.in_bundles_labels:
        bundles_labels.append(nib.load(label).get_fdata())
        bundles_names.append(Path(label).name.split(".")[0])

    if args.bundles_names:
        bundles_names = args.bundles_names

    nb_bundles = len(bundles_labels)
    nb_measures = measures.shape[-1]

    if nb_bundles != measures.shape[-2]:
        measures = np.repeat(measures[:, :, :, np.newaxis, :], nb_bundles, axis=3)
        # parser.error('The number of bundles in the measures does not ' +
        #              'correspond to the bundles labels.')

    nb_labels = int(np.nanmax(bundles_labels[0]) - np.nanmin(bundles_labels[0]))

    density_weights = nib.load(args.in_weights).get_fdata()
    density_weights = np.sum(density_weights, axis=-2)

    mean_profiles = np.zeros((nb_bundles, nb_labels, nb_measures))
    std_profiles = np.zeros((nb_bundles, nb_labels, nb_measures))
    means = np.zeros((nb_bundles, nb_measures))
    stds = np.zeros((nb_bundles, nb_measures))
    for k in range(nb_measures):
        print(measures_name[k])
        for i, bundle in enumerate(bundles_labels):
            print(bundles_names[i])
            masked_data = np.ma.masked_array(measures[..., i, k][bundle >= 1],
                                             np.isnan((measures[..., i, k][bundle >= 1])))
            means[i, k] = np.average(masked_data, weights=density_weights[..., i][bundle >= 1])
            stds[i, k] = np.sqrt(np.average((masked_data - means[i, k]) ** 2, weights=density_weights[..., i][bundle >= 1]))
            # means[i, k] = np.nanmean((measures[..., i, k] * density_weights[... , i])[bundle >= 1])
            # stds[i, k] = np.nanstd((measures[..., i, k] * density_weights[..., i])[bundle >= 1])
            df.loc[(df['Statistics'] == 'mean') & (df['Measures'] == measures_name[k]) & (df['Type'] == args.type) & (df['Bundles'] == bundles_names[i]) & (df['Section'].isnull()), ['Value']] = means[i, k]
            df.loc[(df['Statistics'] == 'std') & (df['Measures'] == measures_name[k]) & (df['Type'] == args.type) & (df['Bundles'] == bundles_names[i]) & (df['Section'].isnull()), ['Value']] = stds[i, k]
            for j in range(nb_labels):
                print("Section ", j + 1)
                masked_data = np.ma.masked_array(measures[..., i, k][bundle == j + 1],
                                             np.isnan((measures[..., i, k][bundle == j + 1])))
                mean_profiles[i, j, k] = np.average(masked_data,
                                                    weights=density_weights[..., i][bundle == j + 1])
                std_profiles[i, j, k] = np.sqrt(np.average((masked_data - mean_profiles[i, j, k]) ** 2,
                                                           weights=density_weights[..., i][bundle == j + 1]))
                # mean_profiles[i, j, k] = np.nanmean((measures[..., i, k] * density_weights[..., i])[bundle == j + 1])
                # std_profiles[i, j, k] = np.nanstd((measures[..., i, k] * density_weights[..., i])[bundle == j + 1])
                df.loc[(df['Statistics'] == 'mean') & (df['Measures'] == measures_name[k]) & (df['Type'] == args.type) & (df['Bundles'] == bundles_names[i]) & (df['Section'] == j + 1), ['Value']] = mean_profiles[i, j, k]
                df.loc[(df['Statistics'] == 'std') & (df['Measures'] == measures_name[k]) & (df['Type'] == args.type) & (df['Bundles'] == bundles_names[i]) & (df['Section'] == j + 1), ['Value']] = std_profiles[i, j, k]

    df.to_csv(args.out_csv)


if __name__ == "__main__":
    main()
