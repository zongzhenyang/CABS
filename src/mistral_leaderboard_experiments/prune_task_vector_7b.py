import torch
from transformers import AutoModelForCausalLM
import argparse
import numpy as np
import copy

def check_sparsity(tensor):
    # Check tensor dtype
    dtype = tensor.dtype
    # Calculate sparsity
    total_elements = tensor.numel()
    zero_elements = torch.sum(tensor == 0).item()
    sparsity = zero_elements / total_elements
    return sparsity, dtype

def check_layer_sparsity(model):
    sparsity_report = {}
    for name, param in model.state_dict().items():
        layer_sparsity, layer_dtype = check_sparsity(param)
        sparsity_report[name] = {
            'layer_sparsity': layer_sparsity,
            'layer_dtype': layer_dtype,
        }
    return sparsity_report

def calculate_overlap(tensor1, tensor2):
    # Check tensor dtype
    dtype1 = tensor1.dtype
    dtype2 = tensor2.dtype
    if dtype1 != dtype2:
        raise ValueError(f"Tensor dtypes do not match: {dtype1} vs {dtype2}")
    # Calculate overlap
    non_zero_tensor1 = tensor1 != 0
    non_zero_tensor2 = tensor2 != 0
    overlap = torch.sum(non_zero_tensor1 & non_zero_tensor2).item()
    total_non_zero = torch.sum(non_zero_tensor1 | non_zero_tensor2).item()
    overlap_ratio = overlap / total_non_zero if total_non_zero > 0 else 0
    return overlap_ratio, dtype1

def check_layer_overlap(model1, model2):
    overlap_report = {}
    for name, param1 in model1.state_dict().items():
        if name in model2.state_dict():
            param2 = model2.state_dict()[name]
            overlap_ratio, dtype = calculate_overlap(param1, param2)
            overlap_report[name] = {
                'overlap_ratio': overlap_ratio,
                'dtype': dtype
            }
    return overlap_report

def nm_pruning(M, n=64, m=128, return_mask=False):
    """
    Apply n:m magnitude pruning to tensor M.
    
    :param M: Input tensor
    :param n: Number of elements to keep in each group
    :param m: Total number of elements in each group
    :param return_mask: Whether to return the mask
    :return: Pruned tensor and mask (if return_mask is True)
    """
    original_shape = M.shape
    M = M.view(-1, m)

    # Get top n elements in each group
    topk_indices = M.abs().topk(n, dim=1).indices
    mask = torch.zeros_like(M).scatter(1, topk_indices, 1)

    pruned_M = M * mask

    if return_mask:
        return pruned_M.view(original_shape), mask.view(original_shape).float()

    return pruned_M.view(original_shape)

def magnitude_pruning(M, sparsity_level=0.5):
    """
    Apply magnitude pruning to tensor M based on a given sparsity level.
    
    :param M: Input tensor
    :param sparsity_level: Proportion of elements to prune
    :return: Pruned tensor
    """
    flattened_param = M.flatten()
    k = int((1 - sparsity_level) * len(flattened_param))
    threshold = torch.topk(flattened_param.abs(), k, largest=False).values.max()
    mask = (M.abs() > threshold).float()
    return M * mask

def sparse_task_vectors(task_vector_model1, task_vector_model2, pruning_method="nm", n=96, m=128, sparsity_level=0.5):
    """
    Apply pruning to task vectors to reduce overlap and maintain target sparsity.
    
    :param task_vector_model1: Task vector model 1
    :param task_vector_model2: Task vector model 2
    :param pruning_method: Pruning method ("nm", "magnitude", "random")
    :param n: Number of elements to keep in each group for n:m pruning
    :param m: Total number of elements in each group for n:m pruning
    :param sparsity_level: Target sparsity level for magnitude or random pruning
    :return: Pruned task vector models 1 and 2
    """
    state_dict1 = task_vector_model1.state_dict()
    state_dict2 = task_vector_model2.state_dict()

    for name in state_dict1.keys():
        if name in state_dict2:
            param1 = state_dict1[name].half()  # Ensure float16
            param2 = state_dict2[name].half()  # Ensure float16

            if pruning_method == "nm":
                # Apply n:m pruning to param1
                param1_pruned, mask1 = nm_pruning(param1, n=n, m=m, return_mask=True)
                
                # Calculate complementary mask for param2
                mask2 = 1 - mask1
                
                # Check if further pruning is needed based on n and m//2 relationship
                if n >= m // 2:
                    param2_remaining = param2 * mask1
                    _, mask_half = nm_pruning(param2_remaining, n=m // 2, m=m, return_mask=True)
                    final_mask2 = torch.clamp(mask2 + mask_half, 0, 1)
                    param2_pruned = param2 * final_mask2
                else:
                    # Directly apply mask2 to param2 to reduce overlap
                    param2_remaining = param2 * mask2
                    param2_pruned, _ = nm_pruning(param2_remaining, n=n, m=m, return_mask=True)
            elif pruning_method == "magnitude":
                # Apply magnitude pruning
                param1_pruned = magnitude_pruning(param1, sparsity_level)
                param2_pruned = magnitude_pruning(param2, sparsity_level)
            elif pruning_method == "random":
                # Apply random pruning based on sparsity level
                mask1 = torch.bernoulli(torch.full(param1.shape, 1 - sparsity_level)).float()
                param1_pruned = param1 * mask1
                mask2 = torch.bernoulli(torch.full(param2.shape, 1 - sparsity_level)).float()
                param2_pruned = param2 * mask2
            else:
                raise ValueError(f"Unsupported pruning method: {pruning_method}")
            
            # Update state dictionaries
            state_dict1[name] = param1_pruned.view(state_dict1[name].shape)  # Ensure shape consistency
            state_dict2[name] = param2_pruned.view(state_dict2[name].shape)  # Ensure shape consistency
    
    # Load pruned parameters back into models
    task_vector_model1.load_state_dict(state_dict1)
    task_vector_model2.load_state_dict(state_dict2)

    return task_vector_model1, task_vector_model2

def apply_task_vectors(base_model, task_vector_model):
    """
    Apply task vectors to the base model.
    
    :param base_model: Base model
    :param task_vector_model: Task vector model
    :return: New model with task vectors applied
    """
    # Create a deep copy of the base model to avoid modifying the original
    new_model = copy.deepcopy(base_model)
    
    for name, param in new_model.named_parameters():
        if name in task_vector_model.state_dict():
            param.data = param.data.half()  # Ensure parameters are float16
            param.data += task_vector_model.state_dict()[name].data.half()
    return new_model

def main():
    parser = argparse.ArgumentParser(description="Apply task vectors to base model and perform pruning.")
    parser.add_argument("--base_model_path", type=str, default="/path/to/base_model/", help="Path to the base model")
    parser.add_argument("--task_vector_path1", type=str, default="/path/to/task_vector_model1/", help="Path to task vector model 1")
    parser.add_argument("--task_vector_path2", type=str, default="/path/to/task_vector_model2/", help="Path to task vector model 2")
    parser.add_argument("--save_directory1", type=str, default="/path/to/save/pruned_model1/", help="Path to save pruned model 1")
    parser.add_argument("--save_directory2", type=str, default="/path/to/save/pruned_model2/", help="Path to save pruned model 2")
    parser.add_argument("--pruning_method", type=str, choices=["nm", "magnitude", "random"], default="nm", help="Pruning method to use")
    parser.add_argument("--n", type=int, default=64, help="Value of n for n:m pruning (required for 'nm' method)")
    parser.add_argument("--m", type=int, default=256, help="Value of m for n:m pruning (required for 'nm' method)")
    parser.add_argument("--sparsity_level", type=float, default=0.75, help="Target sparsity level for magnitude or random pruning")
    args = parser.parse_args()

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.float16)

    # Load task vector models
    task_vector_model1 = AutoModelForCausalLM.from_pretrained(args.task_vector_path1, torch_dtype=torch.float16)
    task_vector_model2 = AutoModelForCausalLM.from_pretrained(args.task_vector_path2, torch_dtype=torch.float16)
    
    # Prune task vectors
    task_vector_model1, task_vector_model2 = sparse_task_vectors(
        task_vector_model1,
        task_vector_model2,
        pruning_method=args.pruning_method,
        n=args.n,
        m=args.m,
        sparsity_level=args.sparsity_level
    )

    # Check sparsity and overlap
    print("Checking sparsity for pruned task vectors:")
    sparsity_report1 = check_layer_sparsity(task_vector_model1)
    sparsity_report2 = check_layer_sparsity(task_vector_model2)
    for layer, report in sparsity_report1.items():
        print(f"Layer: {layer}, Sparsity: {report['layer_sparsity']:.4f}, Dtype: {report['layer_dtype']}")
    for layer, report in sparsity_report2.items():
        print(f"Layer: {layer}, Sparsity: {report['layer_sparsity']:.4f}, Dtype: {report['layer_dtype']}")

    print("\nChecking overlap between pruned task vectors:")
    overlap_report = check_layer_overlap(task_vector_model1, task_vector_model2)
    for layer, report in overlap_report.items():
        print(f"Layer: {layer}, Overlap Ratio: {report['overlap_ratio']:.4f}, Dtype: {report['dtype']}")

    # Apply task vectors to base model
    final_model1 = apply_task_vectors(base_model, task_vector_model1)
    final_model2 = apply_task_vectors(base_model, task_vector_model2)
    
    # Save models
    final_model1.save_pretrained(args.save_directory1)
    final_model2.save_pretrained(args.save_directory2)
    print(f"Saved at {args.save_directory1}")
    print(f"Saved at {args.save_directory2}")

if __name__ == "__main__":
    main()
