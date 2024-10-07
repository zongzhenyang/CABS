# CABS: Conflict-Aware and Balanced Sparsification for Enhancing Model Merging

This repository contains the code for the submission "CABS: Conflict-Aware and Balanced Sparsification for Enhancing Model Merging".

## Abstract

Model merging based on task vectors, i.e., the parameter differences between fine-tuned models and a shared base model, provides an efficient way to integrate multiple models without retraining. This approach can be used to combine task-specific models into a multitask model, improve generalization, or address model deficiencies. One of the significant challenges faced by model merging is the conflicts between task vectors. Existing works aim to mitigate these conflicts through sparsification; however, two issues observed in our experiments significantly limit their performance: *high parameter overlap* and *unbalanced weight distribution*.
To address these issues, we propose a simple yet effective framework called **CABS** (Conflict-Aware and Balanced Sparsification), consisting of **Conflict-Aware Sparsification (CA)** and **Balanced Sparsification (BS)**. CA can reduce parameter overlap by applying masks during sequential pruning, ensuring that each task vector retains distinct, non-overlapping parameters. BS leverages $n$ : $m$ pruning to preserve critical weights while maintaining an even distribution across layers. Our comprehensive experiments demonstrate that CABS outperforms state-of-the-art methods across a range of diverse tasks and model sizes. Notably, in experiments with 7B-parameter language models, CABS surpasses the average performance of an "ideal" model, a virtual model that selects the highest score from individual fine-tuned models for each task (CABS: 76.50 vs. Ideal Model: 76.30 vs. Baseline: 76.02 vs. Fine-tuned Model: 75.86). Our results highlight the importance of addressing both high parameter overlap and unbalanced weight distribution to achieve robust and high-performance model merging.

## Summary figure

![image](https://github.com/user-attachments/assets/9ef9e2bf-d8b3-4a53-bbfc-6fb00f40dcf8)

Illustration of the CABS framework, which enhances model merging by addressing parameter overlap and weight imbalance. By integrating Conflict-Aware Sparsification (CA) and Balanced Sparsification (BS), CABS delivers more effective merging compared to standard merging with magnitude-based pruning (MP), leading to improved model performance.

## Code

### Install Dependencies

To set up the environment, use the provided `environment.yml` file:

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate CABS
```

### Experiments

The repository contains two types of experiments:

1. **RoBERTa GLUE Experiments**: The experiments on RoBERTa are organized under the `roberta_glue_experiments` directory. These experiments include extracting task vectors, sparsifying them, merging, and evaluating the merged models on GLUE tasks.

2. **7B Model Experiments with MergeKit**: The experiments on the 7B parameter model are organized under the `mergekit_7b_experiments` directory. These experiments include using task vectors to perform model merging and evaluating with MergeKit.

Detailed instructions for running the experiments are provided in each respective directory.
