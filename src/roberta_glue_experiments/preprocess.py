import os
from transformers import RobertaTokenizer
from datasets import load_dataset

# Define a global variable to store loaded datasets
LOADED_DATASETS = {}

def preprocess_function(examples, task, tokenizer):
    # Preprocess the input based on the task type
    if task in ["mrpc", "wnli", "rte", "qqp", "stsb"]:
        return tokenizer(examples['sentence1'], examples['sentence2'], padding="max_length", truncation=True, max_length=128)
    elif task in ["mnli"]:
        return tokenizer(examples['premise'], examples['hypothesis'], padding="max_length", truncation=True, max_length=128)
    elif task in ["qnli", "cola", "sst2"]:
        return tokenizer(examples['sentence'], padding="max_length", truncation=True, max_length=128)

def load_and_cache_data(task, tokenizer):
    """Load and cache the dataset. If already loaded, return from cache."""
    if task not in LOADED_DATASETS:
        dataset = load_dataset("glue", task)
        # Preprocess the dataset
        dataset = dataset.map(lambda examples: preprocess_function(examples, task, tokenizer), batched=True)
        LOADED_DATASETS[task] = dataset
    return LOADED_DATASETS[task]
