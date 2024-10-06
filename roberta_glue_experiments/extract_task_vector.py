import torch
import argparse
import copy

def extract_task_vector(finetuned_model_path, base_model_path, save_path):
    """
    Extracts the task-specific vector by subtracting the base model's parameters from the fine-tuned model's parameters.
    Only the differences between the fine-tuned model and the base model are saved.
    
    Args:
        finetuned_model_path (str): Path to the fine-tuned model.
        base_model_path (str): Path to the base model.
        save_path (str): Path to save the extracted task vector.
    """
    # Load models
    finetuned_state_dict = torch.load(finetuned_model_path, map_location='cpu')
    base_state_dict = torch.load(base_model_path, map_location='cpu')
    
    # Create a deepcopy of the fine-tuned model
    task_vector_state_dict = copy.deepcopy(finetuned_state_dict)
    
    # Subtract base model parameters from fine-tuned model parameters
    for name, param in finetuned_state_dict.items():
        if name in base_state_dict:
            task_vector_state_dict[name] = param - base_state_dict[name]
    
    # Save the task vector
    torch.save(task_vector_state_dict, save_path)
    print(f"Task vector has been saved to {save_path}.")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract task vector from fine-tuned model by subtracting base model.")
    parser.add_argument('--finetuned_model_path', type=str, required=True, help="Path to the fine-tuned model.")
    parser.add_argument('--base_model_path', type=str, required=True, help="Path to the base model.")
    parser.add_argument('--save_path', type=str, required=True, help="Path to save the extracted task vector.")
    args = parser.parse_args()

    # Extract task vector
    extract_task_vector(args.finetuned_model_path, args.base_model_path, args.save_path)


# Execute main function
if __name__ == "__main__":
    main()
