#!/bin/bash

# Example script to run model merging and evaluation for RoBERTa GLUE experiments

# Define paths and hyperparameter ranges
MODEL1_PATH="/path/to/rte_vector"
MODEL2_PATH="/path/to/mrpc_vector"
BASE_MODEL_PATH="/path/to/base_model"
SAVE_PATH="/path/to/save/merged_model"
RESULTS_PATH="/path/to/save/results.xlsx"
LAMB1_START=0.4
LAMB1_END=1.0
LAMB1_STEP=0.1
LAMB2_LOWER_OFFSET=0.3
LAMB2_UPPER_OFFSET=0.3
LAMB2_STEP=0.05
TASKS="cola sst2 mrpc rte"

# Run the main.py script with specified arguments
python main.py \
    --lamb1_start $LAMB1_START \
    --lamb1_end $LAMB1_END \
    --lamb1_step $LAMB1_STEP \
    --lamb2_lower_offset $LAMB2_LOWER_OFFSET \
    --lamb2_upper_offset $LAMB2_UPPER_OFFSET \
    --lamb2_step $LAMB2_STEP \
    --model1_path $MODEL1_PATH \
    --model2_path $MODEL2_PATH \
    --base_model_path $BASE_MODEL_PATH \
    --save_path $SAVE_PATH \
    --tasks $TASKS

# Note: Update the paths and hyperparameter ranges as needed for your specific experiment.
