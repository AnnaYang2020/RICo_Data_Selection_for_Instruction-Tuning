"""Select the top classifier-scored samples using the original ranking logic."""

import argparse
import json
import os

import pandas as pd

def main(args):
    if args.target_model == "llama3-8B":
        base_path = "./data_outputs/Llama-3.1-8B/"
        model = 'Llama-3.1-8B'
    if args.target_model=="qwen2.5-3B":
        base_path = "./data_outputs/Qwen2.5-3B/" 
        model = 'Qwen2.5-3B'


    if args.dataset=="Alpaca":
        dataset = "alpaca"
        tot = 52001
    elif args.dataset=="WizardLM":
        dataset = "wizardlm"
        tot = 70000
    num = int(args.percentage/100*tot)
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
    
    if args.score_file:
        file_path = args.score_file
    elif args.dataset=="Alpaca":
        file_path = base_path + 'influence_results/52k_alpaca_merged_influence_results-'+model+"-cl-"+per+'.json'
    elif args.dataset=="WizardLM":
        file_path = base_path + "influence_results/70k_alpaca_merged_wizardLM-influence_results-"+model+"-cl-"+per+'.json'

    
    data = []
    with open(file_path, 'r') as f:
        for line in f:

            try:
                data.append(json.loads(line))
            except:
                print(line)


    df = pd.DataFrame(data)
    df['result2'] = df['result'].apply(lambda x: x[1] if len(x) > 1 else None)


    top_samples = df.nlargest(num, 'result2') #nlargest


    print(top_samples.head())


    output_path = args.output_file or base_path+"training_data/top_"+str(num)+"_samples.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    top_samples.to_json(output_path, orient='records', lines=True)
    print(f"Save to ... {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-model', '--target_model', dest='target_model', default="llama3-8B", type=str)
    parser.add_argument('--dataset', default="Alpaca", type=str)
    parser.add_argument('--percentage', required=True, type=int)
    parser.add_argument('--score-file')
    parser.add_argument('--output-file')
    args = parser.parse_args()
    main(args)
