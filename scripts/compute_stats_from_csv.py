import argparse
import numpy as np
import pandas as pd


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_csv',
                   help='Path of the input CSV file.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)

    if 'Section' in df.columns.tolist():
        df=df[df['Section'].isnull()]
        df.drop('Section', axis=1, inplace=True)

    major_bundles = np.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])

    # Average of the ratio of mean STD between original and corrected
    all_mean_std_ratio = np.zeros((len(df['Bundles'].unique()), 4))
    for i, bundle in enumerate(df['Bundles'].unique()):
        # print(bundle)
        for j, measure in enumerate(['MTR', 'MTsat', 'ihMTR', 'ihMTsat']):
            df_std_ori = df[(df['Statistics'] == 'std') & (df['Measures'] == measure) & (df['Type'] == 'original') & (df['Bundles'] == bundle)]
            df_std_cor = df[(df['Statistics'] == 'std') & (df['Measures'] == measure) & (df['Type'] == 'corrected') & (df['Bundles'] == bundle)]

            mean_std_ori = np.mean(df_std_ori['Value'])
            mean_std_cor = np.mean(df_std_cor['Value'])
            all_mean_std_ratio[i, j] = mean_std_ori / mean_std_cor
            # print(measure, all_mean_std_ratio[i, j])
        # print(np.mean(all_mean_std_ratio[i]))
    print(np.mean(all_mean_std_ratio[major_bundles]))

    # Average of the variation of std of mean between original and corrected
    all_std_mean_ratio = np.zeros((len(df['Bundles'].unique()), 4))
    for i, bundle in enumerate(df['Bundles'].unique()):
        # print(bundle)
        for j, measure in enumerate(['MTR', 'MTsat', 'ihMTR', 'ihMTsat']):
            # print(measure)
            df_mean_ori = df[(df['Statistics'] == 'mean') & (df['Measures'] == measure) & (df['Type'] == 'original') & (df['Bundles'] == bundle)]
            df_mean_cor = df[(df['Statistics'] == 'mean') & (df['Measures'] == measure) & (df['Type'] == 'corrected') & (df['Bundles'] == bundle)]

            std_mean_ori = np.std(df_mean_ori['Value'])
            std_mean_cor = np.std(df_mean_cor['Value'])
            all_std_mean_ratio[i, j] = std_mean_ori / std_mean_cor
            # print(measure, all_std_mean_ratio[i, j])
        # print(np.mean(all_std_mean_ratio[i]))
    print(np.mean(all_std_mean_ratio[major_bundles]))

    # Average of the variation of std of mean between original and corrected per subject
    df[['Subject', 'Session']] = df['Sid'].str.split(pat="_", expand=True, regex=True)
    all_std_mean_ratio = np.zeros((len(df['Subject'].unique()), len(df['Bundles'].unique()), 4))
    for k, sub in enumerate(df['Subject'].unique()):
        # print(sub)
        for i, bundle in enumerate(df['Bundles'].unique()):
            # print(bundle)
            for j, measure in enumerate(['MTR', 'MTsat', 'ihMTR', 'ihMTsat']):
                df_mean_ori = df[(df['Statistics'] == 'mean') & (df['Measures'] == measure) & (df['Type'] == 'original') & (df['Bundles'] == bundle) & (df['Subject'] == sub)]
                df_mean_cor = df[(df['Statistics'] == 'mean') & (df['Measures'] == measure) & (df['Type'] == 'corrected') & (df['Bundles'] == bundle) & (df['Subject'] == sub)]

                std_mean_ori = np.std(df_mean_ori['Value'])
                std_mean_cor = np.std(df_mean_cor['Value'])
                all_std_mean_ratio[k, i, j] = std_mean_ori / std_mean_cor
                # print(measure, all_std_mean_ratio[i, j])
            # print(np.mean(all_std_mean_ratio[i]))
        # print(np.mean(all_std_mean_ratio[k]))
    print(np.mean(all_std_mean_ratio[:, major_bundles]))

if __name__ == "__main__":
    main()
