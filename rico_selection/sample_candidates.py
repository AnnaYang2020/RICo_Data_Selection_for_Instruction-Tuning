"""Sample candidate instructions using the original RICo preprocessing."""

import argparse
import os

import pandas as pd

def main(args):
    PROMPT_DICT = {
        "Alpaca_input": "### Instruction:\n{instruction}\n\n### Input: {input}\n\n### Response:",
        "Alpaca_no_input": "### Instruction:\n{instruction}\n\n### Response:",
    }

    get_data = args.dataset
    cnt = args.quantity

    if args.input_file:
        path = args.input_file
    elif get_data == "Alpaca":
        path = './data/source/alpaca_data.json'
    elif get_data == "Alpaca_GPT4":
        path = './data/source/alpaca_gpt4_data.json'

    if "Alpaca" in get_data:
        df = pd.read_json(path)


    df['dataset'] = get_data
    df['group'] = 'None'
    df = df.rename(columns={'output': 'response'})


    df = df[~(df['instruction'].str.contains('http://', na=False) | df['input'].str.contains('http://', na=False) | df['instruction'].str.contains('https://', na=False) | df['input'].str.contains('https://', na=False))]

    if "Alpaca" in get_data:
        df['inputs'] = df.apply(
            lambda row: PROMPT_DICT['Alpaca_input'].format(**row)
            if pd.notna(row['input']) and row['input'] != '' 
            else PROMPT_DICT['Alpaca_no_input'].format(**row), 
            axis=1
        )

    if len(df) > cnt:
        df_sampled = df.sample(n=cnt, random_state=42)
    else:
        df_sampled = df
        
    df_sampled['output'] = df_sampled['response']
    df_filtered = df_sampled[['dataset', 'inputs', 'response','group', 'instruction', 'input', 'output']]
    output_path = args.output_file or "./data/candidates/000-input_"+get_data+str(cnt*len(df['group'].unique()))+"_sampled_data.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_filtered.to_json(output_path, orient='records', lines=True, force_ascii=False)

    print(f"Sampled data has been saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default="Alpaca", type=str)
    parser.add_argument('--quantity', default=5000, type=int)
    parser.add_argument('--input-file')
    parser.add_argument('--output-file')
    args = parser.parse_args()
    main(args)
