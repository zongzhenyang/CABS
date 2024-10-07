
# 7B Model Experiments Framework

This document describes the framework for conducting experiments on the 7B model using model merging, pruning, and evaluation methodologies. This framework extends similar approaches from the RoBERTa experiments to a larger-scale model.

## Overview

The 7B model experiments are divided into three main steps:

1. **Extract Task-Specific Vector**: Extract task-specific information by subtracting the base model parameters from the fine-tuned model.
2. **Sparsification**: Apply various sparsification methods to the extracted task vectors.
3. **Merging and Evaluation**: Merge the pruned task vectors with the base model and evaluate on specified tasks to understand performance.

## Files and Scripts

- **extract_task_vector_7b.py**: Script for extracting task-specific vectors from the fine-tuned 7B model by subtracting base model parameters.
- **prune_task_vector_7b.py**: Script for applying different sparsification techniques to the task-specific vectors.
- **merge_and_evaluate_7b.py**: Script for merging the pruned vectors with the base model and evaluating performance.
- **scripts/run_7b_experiments.sh**: Bash script to run all the steps in sequence with configurable parameters.

## Running Experiments

### 1. Extract Task-Specific Vector

To extract task-specific vectors, you will need to run the `extract_task_vector_7b.py` script twice with two different fine-tuned models, but the same base model.

#### First Run:

```bash
python extract_task_vector_7b.py \
    --finetuned_model_path BarryFutureman/WildMarcoroni-Variant1-7B \
    --base_model_path Mistral-7b-v0.1 \
    --save_path /path/to/save/task_vector_wildmarcoroni
```

#### Second Run:

```bash
python extract_task_vector_7b.py \
    --finetuned_model_path PetroGPT/WestSeverus-7B-DPO-v2 \
    --base_model_path Mistral-7b-v0.1 \
    --save_path /path/to/save/task_vector_westseverus
```

### 2. Sparsification of Task Vectors

Once the task-specific vectors are extracted, apply sparsification using `prune_task_vector_7b.py`. This needs to be done for both task vectors.

#### For WildMarcoroni-Variant1-7B:

```bash
python prune_task_vector_7b.py \
    --task_vector_path /path/to/save/task_vector_wildmarcoroni \
    --sparsity_level 0.5 \
    --pruning_method "magnitude" \
    --save_path /path/to/save/pruned_vector_wildmarcoroni
```

#### For WestSeverus-7B-DPO-v2:

```bash
python prune_task_vector_7b.py \
    --task_vector_path /path/to/save/task_vector_westseverus \
    --sparsity_level 0.5 \
    --pruning_method "magnitude" \
    --save_path /path/to/save/pruned_vector_westseverus
```

### 3. Model Merging and Evaluation

Finally, merge the pruned task vectors with the base model and evaluate them using the provided recipes for MergeKit and LM-Harness-Evaluation.

- **MergeKit Recipe**: Use MergeKit to merge the pruned vectors.

Run MergeKit with the following command:
```bash
mergekit-yaml /path/to/recipes/recipe.yml /path/to/save/models/model_name/
```

- **LM-Evaluation-Harness**: Use LM-Evaluation-Harness to evaluate the merged model on the specified tasks.

Example command:
```bash
lm-evaluation-harness --model hf --model_args pretrained=/path/to/models --tasks arc_challenge,hellaswag,winogrande,gsm8k --device cuda:0 --batch_size 8 --output_path results.json
```

## Hyperparameters

The hyperparameters for the 7B model experiments are controlled by the following arguments:

- **sparsity_level**: Defines the target sparsity level for pruning.
- **pruning_method**: Specifies the pruning method (magnitude, random, n:m).

## Results

The results of the merging and evaluation step will be saved in a specified output file, containing metrics such as accuracy and task-specific performance.

## Notes

- Make sure to update paths to model files, task vectors, and results as required.
- Adjust hyperparameters to explore their impact on model performance.
```

---
