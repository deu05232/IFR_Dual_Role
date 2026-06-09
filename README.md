# Tevatron (for Promptriever)

This repository is a modified version of the Promptriever fork of Tevatron, with additional support for multi-positive training and improved batch construction strategies.

## Key Features

### Multi-Positive Learning

The following multi-positive training objectives are supported:

- **JointLH**
- **RandLH**

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
pip install -e .

cd src/tevatron/retriever
```

## Training

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

## Evaluation

### InstructIR, FollowIR, MSMARCO, and BEIR

Evaluation follows the original procedures provided by Promptriever. Please refer to the corresponding repositories for dataset preparation and evaluation scripts.

### FollowTable

The newly split FollowTable dataset and evaluation data are provided under the `FollowTable` directory.
