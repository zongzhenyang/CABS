import argparse
import numpy as np
import pandas as pd
from transformers import RobertaTokenizer
from evaluation import evaluate_model_on_task
from model_merge import add_model_params

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run model merging and evaluation for RoBERTa GLUE experiments.")
parser.add_argument('--lamb1_values', type=float, nargs='+', required=True, help="Range of lamb1 values to search.")
parser.add_argument('--model1_path', type=str, required=True, help="Path to model 1.")
parser.add_argument('--model2_path', type=str, required=True, help="Path to model 2.")
parser.add_argument('--save_path', type=str, required=True, help="Path to save the merged model.")
parser.add_argument('--tasks', type=str, nargs='+', required=True, help="Tasks to evaluate.")
args = parser.parse_args()

# Initialize the tokenizer
local_model_path = "/path/to/local/model"
tokenizer = RobertaTokenizer.from_pretrained(local_model_path)

# Define task-specific model paths and label counts
task_specific_paths = {
    "cola": "/path/to/cola/model",
    "sst2": "/path/to/sst2/model",
    "mrpc": "/path/to/mrpc/model",
    "rte": "/path/to/rte/model"
}
num_labels = {"cola": 2, "sst2": 2, "mrpc": 2, "rte": 2, "stsb": 1}  # Set label count based on the task

print("start")

# Define hyperparameter search range and intervals
lamb1_values = np.array(args.lamb1_values)  # Lamb1 values from input arguments
results = []

model1_path = args.model1_path
model2_path = args.model2_path
model3_path = "/path/to/base_model"
save_path = args.save_path
tasks = args.tasks

for lamb1 in lamb1_values:
    lamb2_values = np.arange(lamb1 - 0.8, lamb1 + 0.21, 0.05)
    for lamb2 in lamb2_values:
        # Merge model parameters
        add_model_params(model1_path, model2_path, model3_path, save_path, lamb1, lamb2)
        
        # Evaluate the merged model on each task
        task_performance = {}
        for task in tasks:
            print(f"lamb1={lamb1}, lamb2={lamb2}, Evaluating on {task}...")
            performance = evaluate_model_on_task(task, save_path, task_specific_paths, num_labels, tokenizer)
            if task == 'cola':
                task_performance[task] = performance['accuracy']
            else:
                task_performance[task] = performance['accuracy']
        
        # Save results
        results.append((f"{lamb1}", f"{lamb2}", task_performance))

print("save_result")

# Convert results to DataFrame
data = []
for result in results:
    lamb1, lamb2, performances = result
    row = [lamb1, lamb2] + [performances[task] for task in tasks] + [sum(performances.values()) / len(tasks)]
    data.append(row)

# Create DataFrame from the list
df = pd.DataFrame(data, columns=['lamb1', 'lamb2'] + tasks + ['average'])
# Add model paths to the DataFrame
df['model1_path'] = model1_path
df['model2_path'] = model2_path

# Save DataFrame to Excel file
excel_path = "/path/to/save/results.xlsx"
df.to_excel(excel_path, index=False)

print(f"Results have been saved to {excel_path}")
