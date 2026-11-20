# Tevatron (for Promptriever & Ours)

This repository is a modified version of the Promptriever fork of Tevatron, with additional support for multi-positive training and improved batch construction strategies.

## Key Features

### Multi-Positive Learning

The following multi-positive training objectives are implemented:

- **JointLH:** Jointly optimizes multiple query-level positives in one update (our primary objective).
- **RandLH:** Selects one query-level positive per update.

The experiments also compare **Sample-Add**, which adds an additional training sample instead of jointly optimizing all positives.

### False Negative Prevention Within a Batch

To reduce training noise caused by false negatives, the data loader and batch construction process have been modified to ensure:

- Instances sharing the same query ID (`qid`) are never placed in the same batch.
  - For example, `1234` and `1234-instruct` will not appear together in a batch.
- A positive passage for one instance is never used as a negative passage for another instance within the same batch.

### Validation Support

Validation has been added with support for **nDCG@10** evaluation.

Users can choose whether validation is performed on:

- Original queries (`q`)
- Instruction-augmented queries (`q_inst`)

## Installation

```bash
cd tevatron
pip install -e .

cd src/tevatron/retriever
```

## Training

### Training Data Statistics

| Dataset | # Queries (Q) | # Instruction instances (I) | I/Q | # Documents |
| --- | ---: | ---: | ---: | ---: |
| Instruction-augmented MS MARCO | 489K | 489K | 1.0× | 8.8M |
| FollowTable (WQT) | 240 | 1,345 | 5.6× | 23,784 |


### Instruction-Augmented MSMARCO

To train on the Instruction-Augmented MSMARCO dataset:

```bash
bash train.sh
```

To preserve anonymity during the review process, we withhold the Hugging Face dataset link. The dataset will be publicly released upon acceptance.

### FollowTable

To train on the FollowTable dataset:

```bash
bash train_table.sh
```
### Experimental Configuration

The table summarizes the settings described in the short paper, supplemented with details from the original longer draft and the example scripts.

| Hyperparameter | Text retrieval | FollowTable fine-tuning |
| --- | --- | --- |
| Backbone | Llama 2 7B | Llama 2 7B |
| Fine-tuning | LoRA | LoRA |
| LoRA rank / alpha | 32 / 64 | 32 / 64 |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `down_proj`, `up_proj`, `gate_proj` | Same |
| Learning rate | 1e-4 | 1e-4 |
| Temperature | 0.01 | 0.01 |
| Query maximum length | 304 tokens | 304 tokens |
| Passage/table maximum length | 256 tokens | 512 tokens |
| Candidate group size | 16 | 16 |
| Per-device batch size | 32 | 16 |
| Gradient accumulation steps | 2 | 4 |
| Effective global batch size | 128 | 128 |
| Precision | BF16 | BF16 |
| Representation | EOS pooling, normalized | Same |
| Query / passage prefixes | `query: ` / `passage: ` | Same |
| Query-level positives (JointLH) | 2 | 2 (main), 4 (ablation) |
| Training schedule | 1 epoch; final checkpoint | Up to 40 epochs; best validation checkpoint |
| Seed in example scripts | 42 | 42 |

The reported setup uses two GPUs(H200); effective batch size is per-device batch size × number of GPUs × gradient accumulation steps. The scripts use DeepSpeed ZeRO-3 and FlashAttention 2.


## Evaluation

### Text Retrieval

| Benchmark | Reported metrics |
| --- | --- |
| MS MARCO dev | MRR@10 |
| TREC DL'19 / DL'20 | nDCG@10 |
| BEIR (zero-shot) | nDCG@10 |
| FollowIR | Official Score (effectiveness), p-MRR (instruction sensitivity) |
| InfoSearch (Keyword and Audience) | nDCG@10 (effectiveness), WISE (instruction sensitivity) |
| InstructIR (paraphrased-instruction set) | nDCG@10, Robustness@10 |

Evaluation follows the original procedures provided by Promptriever. Please refer to the corresponding repositories for dataset preparation and evaluation scripts.

### Table Retrieval (Only FollowTable)
The newly split FollowTable dataset and evaluation data are provided under [`FollowTable/`](FollowTable/). We report:

- **Q nDCG@10:** Standard retrieval using original queries.
- **+I nDCG@10:** Instruction-following retrieval using instruction-augmented queries.
- **IRS:** Instruction Responsiveness Score, measuring instruction sensitivity.
