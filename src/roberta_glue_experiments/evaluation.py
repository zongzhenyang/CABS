import torch
import numpy as np
import evaluate
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
from preprocess import load_and_cache_data

def evaluate_model_on_task(task, base_model_path, task_specific_paths, num_labels, tokenizer):
    """Evaluate the model on the specified task, replacing the classifier head with task-specific weights."""
    print(f"Evaluating on {task}...")
    
    # Load the base model
    base_model = RobertaForSequenceClassification.from_pretrained(base_model_path, num_labels=num_labels[task])
    # Load task-specific model to get classifier weights
    if task in task_specific_paths:
        task_model = RobertaForSequenceClassification.from_pretrained(task_specific_paths[task], num_labels=num_labels[task])
        # Get state dictionaries
        base_state_dict = base_model.state_dict()
        task_state_dict = task_model.state_dict()
        # Replace classifier weights
        for key in ["classifier.dense.weight", "classifier.dense.bias", "classifier.out_proj.weight", "classifier.out_proj.bias"]:
            if key in task_state_dict:
                base_state_dict[key] = task_state_dict[key]
        # Load updated state dictionary back to the base model
        base_model.load_state_dict(base_state_dict)
    
    # Load dataset and metrics
    raw_datasets = load_and_cache_data(task, tokenizer)
    metric = evaluate.load('glue', 'sst2') 
    
    # Set training arguments and initialize Trainer
    training_args = TrainingArguments(
        per_device_eval_batch_size=64,
        no_cuda=not torch.cuda.is_available(),
        output_dir=f"./roberta/base-{task}",
    )

    trainer = Trainer(
        model=base_model,
        args=training_args,
        eval_dataset=raw_datasets["validation"],
    )
    
    # Perform evaluation
    predictions = trainer.predict(raw_datasets["validation"]).predictions
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = np.squeeze(predictions)

    # Compute metrics
    metric_score = metric.compute(predictions=predictions, references=raw_datasets["validation"]["label"])
    print(f"{task} score:", metric_score)
    return metric_score
