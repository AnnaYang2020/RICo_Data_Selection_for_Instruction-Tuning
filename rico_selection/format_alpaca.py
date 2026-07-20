"""Convert selected records to Alpaca format using the original field mapping."""

import argparse
import json

import pandas as pd
from tqdm import tqdm


def main(args):
    for file_path in args.file_paths:
        df = pd.read_json(file_path, lines=True)
        gen = []

        for index, row in tqdm(df.iterrows(), desc="Train-data"):
            dict_ = {}
            dict_["instruction"] = row['in_context']["instruction"]
            dict_["input"] = row['in_context']["input"]
            dict_["output"] = row['in_context']["output"]
            gen.append(dict_)

        path = "/".join(file_path.split("/")[:-1])+"/alpaca-format-"+file_path.split("/")[-1]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(gen, f, ensure_ascii=False, indent=4)
        print(path)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file_paths', nargs='+')
    main(parser.parse_args())
