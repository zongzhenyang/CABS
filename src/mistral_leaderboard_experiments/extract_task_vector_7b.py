import torch
import argparse
from transformers import AutoModelForCausalLM

def load_model(model_path):
    """Load the model from the specified path."""
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = model.half()  # Convert model to float16 precision
    return model

def extract_task_vector(base_model, finetuned_model):
    """Extract task-specific vector by subtracting base model parameters from fine-tuned model parameters."""
    task_vector = {}
    for (base_name, base_param), (fine_name, fine_param) in zip(base_model.named_parameters(), finetuned_model.named_parameters()):
        if base_name != fine_name:
            raise ValueError(f"Parameter names do not match: {base_name} vs {fine_name}")
        task_vector[base_name] = fine_param.data - base_param.data
    return task_vector

def save_task_vector(task_vector, save_path):
    """Save the task vector to the specified path using float16 precision."""
    state_dict = {name: param.to(torch.float16) for name, param in task_vector.items()}
    torch.save(state_dict, save_path)
    print(f"Task vector saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract task-specific vector from a fine-tuned model.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base (unfine-tuned) model.")
    parser.add_argument("--finetuned_model_path", type=str, required=True, help="Path to the fine-tuned model.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the extracted task vector.")
    
    args = parser.parse_args()

    # Load models
    print("Loading base model...")
    base_model = load_model(args.base_model_path)
    
    print("Loading fine-tuned model...")
    finetuned_model = load_model(args.finetuned_model_path)

    # Extract task-specific vector
    print("Extracting task-specific vector...")
    task_vector = extract_task_vector(base_model, finetuned_model)

    # Save task-specific vector
    print("Saving task-specific vector...")
    save_task_vector(task_vector, args.save_path)

if __name__ == "__main__":
    main()
