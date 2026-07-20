"""Compute the original random-context control perplexity."""

import torch
import math
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch.multiprocessing as mp
import os
import random
import string
import argparse 

def generate_random_noise(text):

    noise = ''.join(random.choice(string.ascii_letters + string.punctuation + string.digits + ' ') for _ in range(len(text)))
    return noise


def run_inference_on_gpu(gpu_id, data_indices, output_file, custom_dataset1, args):
    if args.target_model == "llama3-8B":
        model_name = "meta-llama/Llama-3.1-8B"
    if args.target_model=="qwen2.5-3B":
        model_name = "Qwen/Qwen2.5-3B"


    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device(f"cuda:{0}")


    model = AutoModelForCausalLM.from_pretrained(model_name).half()
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)


    with open(output_file, 'w') as save_file:
        with torch.no_grad():
            for idx in tqdm(data_indices, desc=f"GPU {gpu_id} Perplexity Calculation"):
                input_text = custom_dataset1[idx]['inputs']
                context_text = custom_dataset1[idx]['context']
                contexy_text_noise = generate_random_noise(context_text)
                target_text = custom_dataset1[idx]['target']
                cat_text = "\n\n\n"


                input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].to(device)
                context_ids = tokenizer(contexy_text_noise, return_tensors="pt", truncation=True, max_length=2040)["input_ids"].to(device)
                target_ids = tokenizer(target_text, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].to(device)
                cat_ids = tokenizer(cat_text, return_tensors="pt", truncation=True, max_length=10)["input_ids"].to(device)
                
                ids = torch.cat([context_ids, cat_ids[:,1:], input_ids[:,1:], target_ids[:,1:]], dim=1).to(device)

                labels = ids.clone()
                input_length = context_ids.size(1) + cat_ids.size(1) + input_ids.size(1)               
                labels[:, :input_length] = -100

                outputs = model(ids, labels=labels)
                loss = outputs.loss

                dict_ = {"in_context": custom_dataset1[idx]['triple1'], "eval_target": custom_dataset1[idx]['triple2']}
                dict_["loss_ctx"] = loss.item()

                dict_['normalized_ppl_ctx'] = math.exp(loss.item()) # ** (1 / length)

                sen_dict = json.dumps(dict_)
                save_file.write(sen_dict + '\n')

    print(f"Process on GPU {gpu_id} completed, results saved to {output_file}.")

def main(args):
    device_ids = [int(i) for i in args.device.split(",")] #[0, 1, 2] #, 4, 5

    if args.target_model == "llama3-8B":
        model_n = 'Llama-3.1-8B'
    if args.target_model=="qwen2.5-3B":
        model_n = 'Qwen-2.5-3B'


    if args.candidate_file:
        file_path1 = args.candidate_file
    elif args.dataset=="Alpaca":  
        file_path1 = "./data/candidates/000-input_Alpaca1_sampled_data.json" #"./data/candidates/000-input_Alpaca5000_sampled_data.json" 
    elif args.dataset=="Alpaca_GPT4":
        file_path1 = "./data/candidates/000-input_Alpaca_GPT45000_sampled_data.json"

    file_1 = open(file_path1, 'r')
    entities1 = file_1.read().strip().split("\n")
    file_1.close()

    file_path2 = args.assessment_file
    file_2 = open(file_path2, 'r')
    entities2 = file_2.read().strip().split("\n")
    file_2.close()

    triples1 = [json.loads(entity1.strip()) for entity1 in entities1]
    triples2 = [json.loads(entity2.strip()) for entity2 in entities2]
    custom_dataset1 = [] 

    for triple1 in triples1:
        for triple2 in triples2:
            custom_dataset1.append({'triple1': triple1, 'triple2': triple2, "inputs": triple2["inputs"], "target": triple2["response"], "context": triple1["inputs"] + triple1["response"]})

    print("Data Preparation Done")



    split_data = torch.chunk(torch.tensor(range(len(custom_dataset1))), len(device_ids))
    mp.set_start_method('spawn', force=True)
    processes = []


    os.makedirs(args.temporary_dir, exist_ok=True)
    output_files = [f"{args.temporary_dir}/output_gpu{device_id}.json" for device_id in device_ids]

    for i, indices in enumerate(split_data):
        indices_list = indices.tolist()
        p = mp.Process(target=run_inference_on_gpu, args=(device_ids[i], indices_list, output_files[i],custom_dataset1, args))
        p.start()
        processes.append(p)


    for p in processes:
        p.join()

    print("All GPU Inference Done, start merging results...")

    output_path = args.output_file or "./data_outputs/"+model_n+"/"+args.dataset+"-ppl_ST_rand.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as merged_file:
        for output_file in output_files:
            with open(output_file, 'r') as infile:
                for line in infile:
                    merged_file.write(line)

    print(f"Results merged into {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', '--devices', dest='device', default="0,1,2", type=str)
    parser.add_argument('--target-model', '--target_model', dest='target_model', default="llama3-8B", type=str)
    parser.add_argument('--dataset', default="Alpaca", type=str)
    parser.add_argument('--candidate-file')
    parser.add_argument('--assessment-file', default='./data/assessment/000-inputs1020_assessment_data.json')
    parser.add_argument('--temporary-dir', default='./data_outputs/tmp/random_context')
    parser.add_argument('--output-file')
    args = parser.parse_args()
    main(args)
