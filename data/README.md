
# Released data

This directory contains the public 1,020-sample assessment set and the currently supplied RICo-selected instruction-tuning subsets. The files are preserved as supplied; dataset filenames and capitalization have not been normalized because they may be referenced by historical experiment commands.

## Assessment set

`assessment/000-inputs1020_assessment_data.json` is a JSON Lines file containing 1,020 records with the fields `dataset`, `inputs`, `response`, and `group`.

As described in the paper, the set is sampled from three public sources to cover ChatGPT-generated, GPT-4-generated, and human-authored instructions:

- OpenOrca-GPT3.5
- OpenOrca-GPT4
- Dolly-15K

The assessment set is used only for assessment and has no overlap with the training data or downstream benchmarks.

## Selected instruction-tuning subsets

All selected-subset files are JSON arrays in Alpaca format with `instruction`, `input`, and `output` fields.

| Selection model | Included subsets | Notes |
| --- | --- | --- |
| LLaMA2-7B | 520, 2,600, 5,200, 7,800 | 1%, 5%, 10%, and 15% filenames include the historical `cl-*-5` suffix |
| LLaMA3.1-8B | 520, 2,600, 5,200, 7,800, 10,400, 13,000, 15,600, 26,000, 39,000 | Top-ranked subsets from 1% through 75% of Alpaca |
| LLaMA3.1-8B | 7,800 | One low-ranked comparison subset |
| Qwen2.5-3B | 520, 2,600, 5,200, 7,800 | Top-ranked 1%, 5%, 10%, and 15% subsets |

The standalone per-sample RICo score files are not present in this directory yet.

## Integrity

SHA-256 checksums for all released JSON files are recorded in [`SHA256SUMS`](SHA256SUMS). Verify them from the repository root with:

```bash
shasum -a 256 -c data/SHA256SUMS
```

## Source dataset links

Exact source-dataset versions and download links will be added once confirmed by the authors.
