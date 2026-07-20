"""Prepare classifier labels with the original quantile and split logic."""

import argparse
import json
import os

import pandas as pd
from tqdm import tqdm

def get_classified_contribution(args):

    if args.score_file:
        file_path = args.score_file
    elif args.target_model=="llama3-8B":
        file_path = "./data_outputs/Llama-3.1-8B/Overall-influence_1020_data-noise-mean-sigmoid.json"
    elif args.target_model=="qwen2.5-3B":
        file_path = "./data_outputs/Qwen2.5-3B/Overall-influence_1020_data-noise-mean-sigmoid.json"

    df = pd.read_json(file_path, lines=True)


    threshold_num = (100-args.percentage)/100
    threshold = df['Significance'].quantile(threshold_num)

    print(threshold_num)


    df['Significance_Mark'] = 0
    df.loc[df['Significance'] >= threshold, 'Significance_Mark'] = 1
    count = df[df['Significance_Mark'] == 1].count()["Significance_Mark"]
    print(count, int(len(df)*(1-threshold_num)))
    

    gen = []
    for index, row in tqdm(df.iterrows(), desc="Train-data"):
        dict_ = {}
        dict_["instruction"] = ''
        dict_["input"] = row['textA']
        dict_["output"] = row['Significance_Mark']
        gen.append(dict_)

    cnt = len(gen)//10
    train_ = gen[:len(gen)-cnt]
    test_ = gen[-cnt:]

    if args.target_model=="llama3-8B":
        base_path = "./data_outputs/Llama-3.1-8B/"
    if args.target_model=="qwen2.5-3B":
        base_path = "./data_outputs/Qwen2.5-3B/"
    if args.output_dir:
        base_path = args.output_dir.rstrip('/') + '/'
    os.makedirs(base_path, exist_ok=True)

    if args.percentage==1:
        per = "1per"
    if args.percentage==5:
        per = "5per"
    if args.percentage==10:
        per = "10per"
    if args.percentage==15:
        per = "15per"
    if args.percentage==20:
        per = "20per"
    if args.percentage==25:
        per = "25per"
    if args.percentage==30:
        per = "30per"
    if args.percentage==50:
        per = "50per"
    if args.percentage==75:
        per = "75per"
    if args.percentage==85:
        per = "85per"

    path = base_path + 'Overall-influence_1020_data-noise-mean-sigmoid-train-cl-'+per+'.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(train_, f, ensure_ascii=False, indent=4)
    path = base_path + 'Overall-influence_1020_data-noise-mean-sigmoid-test-cl-'+per+'.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(test_, f, ensure_ascii=False, indent=4)
    print("Done!")


if __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-model', '--target_model', dest='target_model', default="llama3-8B", type=str)
    parser.add_argument('--percentage', required=True, type=int)
    parser.add_argument('--score-file')
    parser.add_argument('--output-dir')
    args = parser.parse_args()

    get_classified_contribution(args)
