"""Apply the original LoRA selection classifier to a full candidate pool."""

import argparse
import json
import os

import torch
import torch.multiprocessing as mp
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer



def run_inference_on_gpu(gpu_id, data_indices, output_file, target_model, classifier, custom_dataset1, base_model):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device(f"cuda:{0}")


    if base_model:
        model_name = base_model
    elif target_model == "llama3-8B":
        model_name = './model_outputs/Llama-3.1-8B-add'
    elif target_model=="qwen2.5-3B":
        model_name = './model_outputs/Qwen2.5-3B-add'

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, torch_dtype=torch.float16) 
    model = PeftModel.from_pretrained(model, classifier)
    model = model.half()  
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)


    with open(output_file, 'w') as save_file:
        with torch.no_grad():
            for idx in tqdm(data_indices, desc=f"GPU {gpu_id} Perplexity Calculation"):
                input_text = custom_dataset1[idx]['inputs']
                

                input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].to(device)

                outputs = model(input_ids)
                result = torch.nn.functional.softmax(outputs.logits, dim=1).tolist()[0]
                dict_ = {"in_context": custom_dataset1[idx]['triple1'], "result": result}
                
                sen_dict = json.dumps(dict_)
                save_file.write(sen_dict + '\n')

    print(f"Process on GPU {gpu_id} completed, results saved to {output_file}.")

def main(args):
    device_ids = args.device.split(",") #0, 1, 2, , 4, 5


    if args.candidate_file:
        file_path1 = args.candidate_file
    elif args.dataset=="Alpaca":
        file_path1 = "./data/candidates/000-input_Alpaca_all_data.json"
    elif args.dataset=="WizardLM":
        file_path1 = "./data/candidates/000-input_WizardLM_all_sampled_data.json"

    file_1 = open(file_path1, 'r')
    entities1 = file_1.read().strip().split("\n")
    file_1.close()

    triples1 = [json.loads(entity1.strip()) for entity1 in entities1]
    custom_dataset1 = []

    for triple1 in triples1:
        custom_dataset1.append({'triple1': triple1, "inputs": triple1["inputs"] + triple1["response"]})
    print("Data Preparation Done")


    split_data = torch.chunk(torch.tensor(range(len(custom_dataset1))), len(device_ids))

    mp.set_start_method('spawn', force=True)
    processes = []



    if args.target_model == "llama3-8B":
        base_path = "./data_outputs/Llama-3.1-8B/influence_results/"
        model = 'Llama-3.1-8B'
    if args.target_model=="qwen2.5-3B":
        base_path = "./data_outputs/Qwen2.5-3B/influence_results/" 
        model = 'Qwen-2.5-3B'

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
    
    if args.dataset=="Alpaca":
        dataset = "alpaca"
    if args.dataset=="WizardLM":
        dataset = "alpaca"
    
    if args.target_model == "llama3-8B":
        classifier = "./model_outputs/10-lora-"+dataset+"_format-Overall-influence_1020_data-noise-mean-sigmoid-cl-"+per+"-"+model[:-1]
    if args.target_model=="qwen2.5-3B":
        classifier = "./model_outputs/10-lora-"+dataset+"_format-Overall-influence_1020_data-noise-mean-sigmoid-cl-"+per+"-"+"Qwen2.5-3B"
    if args.classifier:
        classifier = args.classifier

    temporary_dir = args.temporary_dir or base_path + 'process'
    os.makedirs(temporary_dir, exist_ok=True)
    output_files = [f"{temporary_dir}/output_gpu{device_id}.json" for device_id in device_ids]

    for i, indices in enumerate(split_data):
        indices_list = indices.tolist()
        p = mp.Process(target=run_inference_on_gpu, args=(device_ids[i], indices_list, output_files[i], args.target_model, classifier,custom_dataset1,args.base_model))
        p.start()
        processes.append(p)


    for p in processes:
        p.join()

    print("All GPU Inference Done, start merging results...")


    

    outputs_path = args.output_file or base_path + '70k_alpaca_merged_wizardLM-influence_results-'+model+"-cl-"+per+'.json'
    if not os.path.exists(base_path):

        os.makedirs(base_path)
    os.makedirs(os.path.dirname(outputs_path), exist_ok=True)
    with open(outputs_path, 'w') as merged_file:
        for output_file in output_files:
            with open(output_file, 'r') as infile:
                for line in infile:
                    merged_file.write(line)

    print(f"Results merged into {outputs_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', '--devices', dest='device', default='0', type=str)
    parser.add_argument('--target-model', '--target_model', dest='target_model', default="llama3-8B", type=str)
    parser.add_argument('--dataset', default="Alpaca", type=str)
    parser.add_argument('--percentage', required=True, type=int)
    parser.add_argument('--candidate-file')
    parser.add_argument('--classifier')
    parser.add_argument('--base-model')
    parser.add_argument('--temporary-dir')
    parser.add_argument('--output-file')
    args = parser.parse_args()
    main(args)
