# RICo Data Selection

Official implementation of **RICo: Refined In-Context Contribution for Automatic Instruction-Tuning Data Selection**.

⭐⭐⭐ **Accepted to the 40th AAAI Conference on Artificial Intelligence (AAAI 2026).** ⭐⭐⭐

RICo is a gradient-free data-selection method that estimates the contribution of instruction-tuning samples through in-context learning. It measures task-level effects on a diverse assessment set, aggregates them into a global RICo score, and trains a lightweight classifier to scale selection to a full candidate pool.

- Paper: [AAAI Proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/40732) | [arXiv:2505.05327](https://arxiv.org/abs/2505.05327)
- Paper arXiv: [arXiv:2505.05327](https://arxiv.org/abs/2505.05327)
- Project page: [annayang2020.github.io/RICo_Data_Selection](https://annayang2020.github.io/RICo_Data_Selection/)
- Released data: [`data/README.md`](data/README.md)

## Repository layout

```text
RICo/
├── rico_selection/       # RICo scoring and scalable data selection
├── training/             # Classifier and instruction-tuning code
│   └── config/           # DeepSpeed configuration
├── data/                 # Assessment set and released selected subsets
└── requirements.txt
```

## Environment

The paper reports the following experiment environment:

- Ubuntu 20.04
- NVIDIA A40 GPUs and 502 GiB system RAM
- Python 3.9.19
- PyTorch 2.4.0
- CUDA 12.1

Create an isolated environment:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The exact historical versions of Transformers, PEFT, Accelerate, DeepSpeed, pandas, and other packages are unavailable. `requirements.txt` therefore pins the reported PyTorch version and lists the remaining direct dependencies without speculative version pins.

LLaMA 3.1 is gated on Hugging Face. Request model access and authenticate before running commands that use `meta-llama/Llama-3.1-8B`.

## RICo scoring pipeline

Run commands from the repository root. The examples below use LLaMA3.1-8B, Alpaca, and GPUs `0,1`. Replace paths and devices as needed.

### 1. Sample candidate instructions

Place the source dataset at `data/source/alpaca_data.json`, or pass another path with `--input-file`.

```bash
python rico_selection/sample_candidates.py \
    --dataset Alpaca \
    --quantity 5000
```

The default output is `data/candidates/000-input_Alpaca5000_sampled_data.json`.

### 2. Compute baseline assessment perplexity

```bash
python rico_selection/compute_assessment_ppl.py \
    --device 0 \
    --target-model llama3-8B
```

This stage computes `PPL(S)`: the target model's perplexity on each assessment sample without a candidate demonstration. It is required as the normalization denominator in the RICo influence score. 

### 3. Compute candidate-conditioned perplexity

```bash
python rico_selection/compute_context_ppl.py \
    --devices 0,1 \
    --target-model llama3-8B \
    --dataset Alpaca \
    --candidate-file data/candidates/000-input_Alpaca5000_sampled_data.json
```

This stage computes `PPL(S|T)` for every candidate-assessment pair.

### 4. Compute the random-context control

```bash
python rico_selection/compute_random_context_ppl.py \
    --devices 0,1 \
    --target-model llama3-8B \
    --dataset Alpaca \
    --candidate-file data/candidates/000-input_Alpaca5000_sampled_data.json
```

This stage replaces each candidate context with a length-matched random context and computes the control perplexity.

### 5. Aggregate global RICo scores

```bash
python rico_selection/compute_rico_scores.py \
    --target-model llama3-8B \
    --dataset Alpaca
```

The default output is `data_outputs/Llama-3.1-8B/global_rico_Alpaca.json`.

## Train and use the selection classifier

### 6. Prepare classifier train/test data

The following command labels the top 15% of sampled candidates as positive examples and preserves the original 90/10 sequential split:

```bash
python rico_selection/prepare_classifier_data.py \
    --target-model llama3-8B \
    --percentage 15 \
    --score-file data_outputs/Llama-3.1-8B/global_rico_Alpaca.json \
    --output-dir data_outputs/Llama-3.1-8B/classifier_data
```

### 7. Train the LoRA selection classifier

```bash
bash training/run_train_classifier.sh \
    data_outputs/Llama-3.1-8B/classifier_data/Overall-influence_1020_data-noise-mean-sigmoid-train-cl-15per.json \
    data_outputs/Llama-3.1-8B/classifier_data/Overall-influence_1020_data-noise-mean-sigmoid-test-cl-15per.json \
    model_outputs/rico-classifier-llama3.1-8b-15per \
    meta-llama/Llama-3.1-8B
```

The classifier uses 5 epochs, learning rate `5e-5`, LoRA rank 8, and LoRA alpha 8.

### 8. Score the full candidate pool

The full-pool input is JSONL with the same `inputs` and `response` fields produced by `sample_candidates.py`.

```bash
python rico_selection/score_candidates.py \
    --devices 0,1 \
    --target-model llama3-8B \
    --dataset Alpaca \
    --percentage 15 \
    --candidate-file data/candidates/000-input_Alpaca_all_data.json \
    --classifier model_outputs/rico-classifier-llama3.1-8b-15per \
    --base-model meta-llama/Llama-3.1-8B \
    --output-file data_outputs/Llama-3.1-8B/selection_results/Alpaca-15per.json
```

### 9. Select and format the top candidates

```bash
python rico_selection/select_top_samples.py \
    --target-model llama3-8B \
    --dataset Alpaca \
    --percentage 15 \
    --score-file data_outputs/Llama-3.1-8B/selection_results/Alpaca-15per.json

python rico_selection/format_alpaca.py \
    data_outputs/Llama-3.1-8B/training_data/top_7800_samples.json
```

## Instruction tuning

Train the target model on an Alpaca-format selected subset:

```bash
bash training/run_instruction_tuning.sh \
    data/llama3.1-8B/alpaca-format-top_7800_samples.json \
    model_outputs/llama3.1-8b-rico-15per \
    meta-llama/Llama-3.1-8B
```

The provided wrapper uses 3 epochs, AdamW, a learning rate of `2e-5`, a per-device batch size of 4, 8 gradient-accumulation steps, and DeepSpeed offloading.

## Code overview

| File | Purpose |
| --- | --- |
| `rico_selection/sample_candidates.py` | Candidate sampling and prompt formatting |
| `rico_selection/compute_assessment_ppl.py` | Baseline `PPL(S)` |
| `rico_selection/compute_context_ppl.py` | Candidate-conditioned `PPL(S\|T)` |
| `rico_selection/compute_random_context_ppl.py` | Random-context control |
| `rico_selection/compute_rico_scores.py` | Global RICo score aggregation |
| `rico_selection/prepare_classifier_data.py` | Classifier label/data preparation |
| `rico_selection/score_candidates.py` | Full-pool classifier scoring |
| `rico_selection/select_top_samples.py` | Top-score selection |
| `rico_selection/format_alpaca.py` | Alpaca-format conversion |
| `training/train_classifier.py` | LoRA selection-classifier training |
| `training/train.py` | Final instruction tuning |

## Released data

The repository contains the final 1,020-sample assessment set and currently supplied selected subsets for LLaMA2-7B, LLaMA3.1-8B, and Qwen2.5-3B. See [`data/README.md`](data/README.md) for sample counts and formats.

## Code attribution

The training implementation in `training/train.py` and the adapted classifier-training implementation in `training/train_classifier.py` are based on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) training code. The original copyright and Apache License 2.0 notices are retained in both files.

## Citation

```bibtex
@inproceedings{yang2026rico,
  title={RICo: Refined In-Context Contribution for Automatic Instruction-Tuning Data Selection},
  author={Yang, Yixin and Dong, Qingxiu and Yao, Linli and Zhu, Fangwei and Luo, Weilin and Wang, Bin and Sui, Zhifang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={40},
  pages={34349--34357},
  year={2026}
}
```
