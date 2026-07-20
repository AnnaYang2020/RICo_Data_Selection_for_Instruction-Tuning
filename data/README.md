# Released data

This directory contains the public 1,020-sample assessment set and the currently supplied RICo-selected instruction-tuning subsets.

## Assessment set

`assessment/000-inputs1020_assessment_data.json` is a JSON Lines file containing 1,020 records with the fields `dataset`, `inputs`, `response`, and `group`.

As described in the paper, the set is sampled from three public sources to cover ChatGPT-generated, GPT-4-generated, and human-authored instructions:

- OpenOrca-GPT3.5
- OpenOrca-GPT4
- Dolly-15K

The assessment set is used only for assessment and has no overlap with the training data or downstream benchmarks.

## Selected instruction-tuning subsets

All RICo-selected subset files are JSON arrays in Alpaca format with `instruction`, `input`, and `output` fields.

| Selection model | Included subsets | Notes |
| --- | --- | --- |
| LLaMA2-7B | 520, 2,600, 5,200, 7,800 | Top-ranked 1%, 5%, 10%, and 15% subsets |
| LLaMA3.1-8B | 520, 2,600, 5,200, 7,800, 10,400, 13,000, 15,600, 26,000, 39,000 | Top-ranked subsets from 1% through 75% of Alpaca |
| LLaMA3.1-8B | 7,800 | One low-ranked comparison subset |
| Qwen2.5-3B | 520, 2,600, 5,200, 7,800 | Top-ranked 1%, 5%, 10%, and 15% subsets |

## Source dataset links

- Alpaca: [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)
- WizardLM-70K: [WizardLMTeam/WizardLM_evol_instruct_70k](https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_70k)
- OpenOrca GPT-3.5/GPT-4: [Open-Orca/OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca)
- Dolly-15K: [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
