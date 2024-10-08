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

Once the task-specific vectors are extracted, apply sparsification to both vectors simultaneously using `prune_task_vector_7b.py`.

```bash
python prune_task_vector_7b.py \
    --task_vector_path1 /path/to/task_vector_wildmarcoroni \
    --task_vector_path2 /path/to/task_vector_westseverus \
    --n 64 \
    --m 256 \
    --pruning_method "nm" \
    --sparsity_level 0.75 \
    --save_directory1 /path/to/save/pruned_vector_wildmarcoroni \
    --save_directory2 /path/to/save/pruned_vector_westseverus
```

### 3. Model Merging and Evaluation

Finally, merge the pruned task vectors with the base model and evaluate them using the provided recipes for MergeKit and LM-Harness-Evaluation.

- **MergeKit Recipe**: Use MergeKit to merge the pruned vectors.

Run MergeKit with the following command:
```
mergekit-yaml /path/to/recipes/recipe.yml /path/to/save/models/model_name/
```

- **LM-Evaluation-Harness**: Use LM-Evaluation-Harness to evaluate the merged model on the specified tasks.

Example command:
```
lm-evaluation-harness --model hf --model_args pretrained=/path/to/models/model_name --tasks arc_challenge,hellaswag,truthfulqa_mc2,winogrande,gsm8k,mmlu --device cuda:0 --batch_size 8 --output_path results.json
```

## Hyperparameters

The hyperparameters for the 7B model experiments are controlled by the following arguments:

- **n**: Number of elements to keep in each group for pruning (required for `nm` pruning).
- **m**: Total number of elements in each group for pruning (required for `nm` pruning).
- **sparsity_level**: Defines the target sparsity level for pruning (required for `random` and `magnitude` pruning).
- **pruning_method**: Specifies the pruning method (`nm`, `random`, `magnitude`).

## Results

The results of the merging and evaluation step will be saved in a specified output file, containing metrics such as accuracy and task-specific performance.

## Notes

- Make sure to update paths to model files, task vectors, and results as required.
- Adjust hyperparameters to explore their impact on model performance.
