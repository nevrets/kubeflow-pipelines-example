import pandas as pd
import argparse

import logging

if __name__ == "__main__":
    
    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument(
        '--data_path', type=str,
        help="Input data path"
    )

    args = argument_parser.parse_args()
    data = pd.read_csv(args.data_path)
    print(data.shape)

    logging.info(data.head())
    
    print("load data")

    data.to_csv('/Iris.csv', index=False)