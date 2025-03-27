import argparse
import pandas as pd


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_csv',
                   help='Path of the input CSV file.')
    p.add_argument('out_csv',
                   help='Path of the output CSV file.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)

    df.rename(columns={'sid': 'Sid', 'roi': 'Bundles', 'metrics': 'Measures', 'value': 'Value', 'section': 'Section', 'stats': 'Statistics'}, inplace=True)
    df.drop('Unnamed: 0', axis =1, inplace=True)

    df['Measures'] = df['Measures'].replace('_metric', '', regex=True)

    df[['Measures', 'Type']] = df['Measures'].str.split(pat="_", expand=True, regex=True)

    df.to_csv(args.out_csv)


if __name__ == "__main__":
    main()
