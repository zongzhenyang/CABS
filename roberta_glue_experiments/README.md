# RoBERTa GLUE Experiments

This repository contains the code and scripts for performing model merging, evaluation, and sparsification experiments using the RoBERTa model on GLUE tasks. The following sections provide an overview of how to use the scripts for different tasks.

## Overview

The experiments involve merging models, pruning model parameters, and evaluating their performance on various GLUE tasks. The main steps include:

1. **Extract Task-Specific Vector**: Extract task-specific vectors by subtracting base model parameters from fine-tuned model parameters.
2. **Sparsification of Task Vectors**: Apply different sparsification methods to reduce the size of the extracted task vectors.
3. **Merging Models and Evaluation**: Merge the pruned task vectors with the base model and evaluate the resulting models on the GLUE benchmark to measure performance metrics.

## Files and Scripts

- **`main.py`**: Main script to merge models and evaluate them on GLUE tasks.
- **`model_merge.py`**: Contains functions to merge parameters from multiple models.
- **`evaluation.py`**: Evaluates merged models on specified GLUE tasks.
- **`preprocess.py`**: Preprocesses the dataset to prepare it for evaluation.
- **`extract_task_vector.py`**: Extracts the task-specific vector by subtracting the base model's parameters from the fine-tuned model's parameters.
- **`scripts/run_merge_and_evaluate.sh`**: Bash script to run model merging and evaluation with predefined hyperparameter ranges.

## Prerequisites

- Python 3.7 or above
- PyTorch
- Hugging Face Transformers library
- Required Python packages (can be installed using `requirements.txt`)

## Running Experiments

### 1. Model Merging and Evaluation

The main merging and evaluation can be executed using the script `scripts/run_merge_and_evaluate.sh`. Update the paths and hyperparameter values in the script as per your requirement.

```bash
bash scripts/run_merge_and_evaluate.sh
```

### 2. Extract Task-Specific Vector

To extract the task-specific vector from a fine-tuned model, run the `extract_task_vector.py` script with the required arguments:

```bash
python extract_task_vector.py \
    --finetuned_model_path /path/to/finetuned_model \
    --base_model_path /path/to/base_model \
    --save_path /path/to/save/task_vector
```

### 3. Sparsification of Task Vectors

The script `main.py` can be used to sparsify the model parameters after merging. The sparsification step uses various pruning strategies such as magnitude pruning, random pruning, and n:m pruning. The parameters for pruning can be adjusted in the script or through command-line arguments.

## Hyperparameters

The hyperparameters for model merging and evaluation are controlled by the following arguments:

- **`lamb1_start`, `lamb1_end`, `lamb1_step`**: Control the range for the first merging coefficient (`lamb1`).
- **`lamb2_lower_offset`, `lamb2_upper_offset`, `lamb2_step`**: Control the range for the second merging coefficient (`lamb2`) relative to `lamb1`.
- **`sparsity`**: Specifies the target sparsity level for pruning.

## Results

The results of the model merging and evaluation are saved as an Excel file at the specified path (`RESULTS_PATH`). This file contains the performance metrics for each task, including accuracy and average performance.

## Notes

- Ensure that the paths specified in the scripts (`model paths`, `save paths`, etc.) are updated to match your directory structure.
- Modify the hyperparameter values to explore different merging strategies and observe their impact on performance.

## License

This repository is licensed under the MIT License.

