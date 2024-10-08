# RoBERTa GLUE Experiments

This repository contains the code and scripts for performing model merging, evaluation, and sparsification experiments using the RoBERTa model on GLUE tasks. The following sections provide an overview of how to use the scripts for different tasks.

## Overview

The experiments involve merging models, pruning model parameters, and evaluating their performance on various GLUE tasks. The main steps include:

1. **Extract Task-Specific Vector**: Extract task-specific vectors by subtracting base model parameters from fine-tuned model parameters.
2. **Sparsification of Task Vectors**: Apply different sparsification methods to reduce the size of the extracted task vectors.
3. **Merging Models and Evaluation**: Merge the pruned task vectors with the base model and evaluate the resulting models on the GLUE benchmark to measure performance metrics.

**Models**: For each task, we utilized pre-trained and fine-tuned versions of RoBERTa, obtained from Hugging Face. Specifically, we used [FacebookAI/roberta-base](https://huggingface.co/facebook/roberta-base) as the base model. Fine-tuned models include [textattack/roberta-base-CoLA](https://huggingface.co/textattack/roberta-base-CoLA), [textattack/roberta-base-SST-2](https://huggingface.co/textattack/roberta-base-SST-2), [textattack/roberta-base-MRPC](https://huggingface.co/textattack/roberta-base-MRPC), and [textattack/roberta-base-RTE](https://huggingface.co/textattack/roberta-base-RTE).

## Files and Scripts

- **`main.py`**: Main script to merge models and evaluate them on GLUE tasks.
- **`model_merge.py`**: Contains functions to merge parameters from multiple models.
- **`evaluation.py`**: Evaluates merged models on specified GLUE tasks.
- **`preprocess.py`**: Preprocesses the dataset to prepare it for evaluation.
- **`extract_task_vector.py`**: Extracts the task-specific vector by subtracting the base model's parameters from the fine-tuned model's parameters.
- **`scripts/run_merge_and_evaluate.sh`**: Bash script to run model merging and evaluation with predefined hyperparameter ranges.

## Running Experiments

### 1. Extract Task-Specific Vector

To extract the task-specific vector from a fine-tuned model, run the `extract_task_vector.py` script with the required arguments:

```bash
python extract_task_vector.py \
    --finetuned_model_path /path/to/finetuned_model \
    --base_model_path /path/to/base_model \
    --save_path /path/to/save/task_vector
```

### 2. Sparsification of Task Vectors

To sparsify the extracted task vectors, use the `main.py` script with appropriate arguments to apply different pruning methods. The sparsification step can use magnitude pruning, random pruning, or n:m pruning. Example command:

```bash
python prune_task_vector.py \
    --model1_path /path/to/task_vector1 \
    --model2_path /path/to/task_vector2 \
    --save_path1 /path/to/save/pruned_model1 \
    --save_path2 /path/to/save/pruned_model2 \
    --pruning_method magnitude \
    --sparsity_level 0.90625
```

For n:m pruning, specify the `n` and `m` values:

```bash
python prune_task_vector.py \
    --model1_path /path/to/task_vector1 \
    --model2_path /path/to/task_vector2 \
    --save_path1 /path/to/save/pruned_model1 \
    --save_path2 /path/to/save/pruned_model2 \
    --pruning_method n:m \
    --n 3 \
    --m 32
```

To merge and evaluate models with different hyperparameter ranges, use the following command:

```bash
python main.py \
    --model1_path /path/to/task_vector1 \
    --model2_path /path/to/task_vector2 \
    --base_model_path /path/to/base_model \
    --save_path /path/to/save/pruned_model \
    --lamb1_start 0.4 \
    --lamb1_end 0.9 \
    --lamb1_step 0.1 \
    --lamb2_lower_offset 0.20 \
    --lamb2_upper_offset 0.20 \
    --lamb2_step 0.1 \
    --tasks mrpc+rte
```

### 3. Model Merging and Evaluation

The main merging and evaluation can be executed using the script `scripts/run_merge_and_evaluate.sh`. Update the paths and hyperparameter values in the script as per your requirement.

```bash
bash scripts/run_merge_and_evaluate.sh
```

## Hyperparameters

The hyperparameters for model merging and evaluation are controlled by the following arguments:

- **`pruning_method`**: Specifies the pruning method (`magnitude`, `random`, `n:m`).
- **`sparsity_level`**: Specifies the target sparsity level for magnitude or random pruning.
- **`n`, `m`**: Parameters for n:m pruning, specifying how many elements to keep (`n`) in each group of size (`m`).
- **`lamb1_start`, `lamb1_end`, `lamb1_step`**: Control the range for the first merging coefficient (`lamb1`).
- **`lamb2_lower_offset`, `lamb2_upper_offset`, `lamb2_step`**: Control the range for the second merging coefficient (`lamb2`) relative to `lamb1`.

## Results

The results of the model merging and evaluation are saved as an Excel file at the specified path (`RESULTS_PATH`). This file contains the performance metrics for each task, including accuracy and average performance.

## Notes

- Ensure that the paths specified in the scripts (`model paths`, `save paths`, etc.) are updated to match your directory structure.
- Modify the hyperparameter values to explore different merging strategies and observe their impact on performance.

## License

This repository is licensed under the MIT License.
