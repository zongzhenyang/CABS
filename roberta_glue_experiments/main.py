import numpy as np
import pandas as pd
from transformers import RobertaTokenizer
from evaluation import evaluate_model_on_task
from model_merge import add_model_params

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
lamb1_values = np.arange(9, 11.51, 0.05)  # Example: from 9 to 11.5 with step size 0.05
results = []

model1_path = "/path/to/model1"
model2_path = "/path/to/model2"
model3_path = "/path/to/base_model"
save_path = "/path/to/save/merged_model"
tasks = ["cola", "sst2"]

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
