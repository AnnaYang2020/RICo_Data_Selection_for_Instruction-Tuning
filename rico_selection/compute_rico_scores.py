"""Aggregate the original three perplexity components into RICo scores."""

import argparse
import math
import os

import pandas as pd

def sigmoid_map(x, scale=50):
    if x>0.5:
        return 1.0
    if x<-0.5:
        return -1.0
    return 2 / (1 + math.exp(-scale * x)) - 1


def main(args):

    if args.target_model == "llama3-8B":
        base_path = "./data_outputs/Llama-3.1-8B/influence_results/"
        model = 'Llama-3.1-8B'
    if args.target_model=="qwen2.5-3B":
        base_path = "./data_outputs/Qwen2.5-3B/influence_results/" 
        model = 'Qwen-2.5-3B'
    
    if args.dataset=="Alpaca":
        dataset = "alpaca"
    if args.dataset=="Alpaca_GPT4":
        dataset = "alpaca_gpt4"

    print("Start!")
    file_path1 = args.context_file or "./data_outputs/"+model+"/"+args.dataset+"-ppl_ST.json"
    file_path2 = args.baseline_file or "./data_outputs/"+model+"/ppl_S.json"
    file_path3 = args.random_context_file or "./data_outputs/"+model+"/"+args.dataset+"-ppl_ST_rand.json"

    output_path = args.output_file or "./data_outputs/"+model+"/global_rico_"+args.dataset+".json"
    print("Loading dataset...")
    df1 = pd.read_json(file_path1, lines=True)
    df2 = pd.read_json(file_path2, lines=True)
    df3 = pd.read_json(file_path3, lines=True)
    print("Done")
    df2['eval_target_key1'] = df2['eval_target'].apply(lambda x: x.get('inputs', None))
    df2['eval_target_key2'] = df2['eval_target'].apply(lambda x: x.get('response', None))
    df1['eval_target_key1'] = df1['eval_target'].apply(lambda x: x.get('inputs', None))
    df1['eval_target_key2'] = df1['eval_target'].apply(lambda x: x.get('response', None))
    df3['eval_target_key1'] = df3['eval_target'].apply(lambda x: x.get('inputs', None))
    df3['eval_target_key2'] = df3['eval_target'].apply(lambda x: x.get('response', None))
    df3 = df3.rename(columns={'loss_ctx': 'loss_ctx_noise', 'normalized_ppl_ctx':'normalized_ppl_ctx_noise'})
    df1 = pd.merge(df1, df2, on=['eval_target_key1', 'eval_target_key2'], how="left")
    df1 = df1.rename(columns={'in_context': 'in_context_x', 'eval_target_key1': 'eval_target_key1_x', 'eval_target_key2': 'eval_target_key2_x'})
    df3 = pd.concat([df1, df3], axis=1)
    is_eval_equal = df3['eval_target_key1_x'] == df3['eval_target_key1']
    assert is_eval_equal.all(), "eval columns are not equal in all rows!"
    print("Concat...Done")

    print("Geting influence score...")
    df3["textA"] = df3["in_context"].apply(lambda x: x.get('inputs', None) + x.get('response', None))
    df3["textB"] = df3['eval_target_y'].apply(lambda x: x.get('inputs', None) + x.get('response', None))
    df3["influence"] = df3.apply(
        lambda row: 
            (-row["normalized_ppl_ctx"]+row["normalized_ppl_ctx_noise"]) / (row["normalized_ppl_ori"] + 0.000001),
        axis=1
    )
    print("Done")

    df_filtered = df3[['in_context', 'eval_target', 'textA','textB', 'loss_ori','loss_ctx', 'loss_ctx_noise','influence']]

    print("Geting overall scores...")
    Significance = df_filtered.groupby('textA').apply(lambda x: sigmoid_map(x['influence'].mean()), include_groups=False)
    Consistency = df_filtered.groupby('textA').apply(lambda x: (x['influence'] > 0).sum() / len(x['influence']), include_groups=False)
    Stability = df_filtered.groupby('textA')['influence'].var()
    Coverage = df_filtered.groupby('textA').apply(lambda x: (x['influence'] > 0.5).sum() / len(x['influence']), include_groups=False)

    df5 = df_filtered[['in_context', 'textA']].drop_duplicates('textA').set_index('textA')
    df5['Significance'] = Significance
    df5['Consistency'] = Consistency
    df5['Stability'] = Stability
    df5['Coverage'] = Coverage
    print("Done")

    df5 = df5.reset_index()

    print("Saving results...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df5.to_json(output_path, orient='records', lines=True, force_ascii=False)

    print(f"Results has been saved to {output_path}")
    print("Result sample:\n", df_filtered.loc[0])

    pair_output_path = args.pair_output_file or "./data_outputs/"+model+"/rico_pair_scores_"+args.dataset+".json"
    df3.to_json(pair_output_path, orient='records', lines=True, force_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-model', '--target_model', dest='target_model', default="llama3-8B", type=str)
    parser.add_argument('--dataset', default="Alpaca", type=str)
    parser.add_argument('--context-file')
    parser.add_argument('--baseline-file')
    parser.add_argument('--random-context-file')
    parser.add_argument('--output-file')
    parser.add_argument('--pair-output-file')
    args = parser.parse_args()
    main(args)
