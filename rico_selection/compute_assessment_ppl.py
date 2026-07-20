"""Compute the original baseline assessment perplexity PPL(S)."""

import argparse
import json
import math
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_perplexity(args): 

    rank = args.device
    device = torch.device(f"cuda:{rank}")

    if args.target_model == "llama3-8B":
        model_name = "meta-llama/Llama-3.1-8B"
    if args.target_model == "qwen2.5-3B":
        model_name = "Qwen/Qwen2.5-3B"
    model = AutoModelForCausalLM.from_pretrained(model_name).half()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)


    file_path2 = args.assessment_file
    file_2 = open(file_path2, 'r')
    entities2 = file_2.read().strip().split("\n")
    file_2.close()

    if args.output_file:
        output_file = args.output_file
    elif args.target_model == "llama3-8B":
        output_file = './data_outputs/Llama-3.1-8B/ppl_S.json'
    elif args.target_model == "qwen2.5-3B":
        output_file = './data_outputs/Qwen-2.5-3B/ppl_S.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_file = open(output_file, 'a')

    triples2 = [json.loads(entity2.strip()) for entity2 in entities2]
    custom_dataset1 = [] 
    custom_dataset2 = [] 
    
    for triple2 in triples2:
        custom_dataset2.append({'triple2': triple2, "inputs": triple2["inputs"], "target": triple2["response"]})

    model.eval()
    with torch.no_grad():

        for i in tqdm(range(len(custom_dataset2)), desc="Perplexity Calculation"):
            input_text = custom_dataset2[i]['inputs']
            target_text = custom_dataset2[i]['target']
            
            input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].to(device)
            target_ids = tokenizer(target_text, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].to(device)
                
            ids = torch.cat([input_ids, target_ids[:,1:]], dim=1).to(device)

            labels = ids.clone()
            input_length = input_ids.size(1)
            labels[:, :input_length] = -100 
            
            outputs = model(ids, labels=labels)
            loss = outputs.loss

            dict_ = {"eval_target": custom_dataset2[i]['triple2']}
            dict_["loss_ori"] = loss.item()

            dict_['normalized_ppl_ori'] = math.exp(loss.item()) # ** (1 / length)

            sen_dict = json.dumps(dict_)
            save_file.write(sen_dict + '\n')

    save_file.close()

def main(args):
    compute_perplexity(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default=0, type=int, help='CUDA device index')
    parser.add_argument('--target-model', '--target_model', dest='target_model', default="llama3-8B", choices=['llama3-8B', 'qwen2.5-3B'])
    parser.add_argument('--assessment-file', default='./data/assessment/000-inputs1020_assessment_data.json')
    parser.add_argument('--output-file')
    args = parser.parse_args()
    main(args)
            
