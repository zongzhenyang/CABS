import torch 
import argparse


def topk_values_mask(param, sparsity_level=0.5, return_mask=False):
    """Applies top-K pruning to the parameter tensor based on sparsity level."""
    flattened_param = param.flatten()
    sorted_tensor = torch.sort(flattened_param.abs(), descending=False)[0]
    threshold_index = int(sparsity_level * len(sorted_tensor))
    threshold = sorted_tensor[threshold_index]
    mask = (param.abs() > threshold).float()

    sparse_param = param * mask

    if return_mask:
        return sparse_param, mask
    else:
        return sparse_param


def n_m_block_pruning(param, n, m):
    """Applies n:m sparsity to the parameter tensor."""
    param = param.clone()  # To avoid modifying the original tensor in place
    mask = torch.zeros_like(param)
    flat_param = param.flatten()

    for i in range(0, flat_param.size(0), m):
        block = flat_param[i:i + m]
        if block.size(0) < m:
            break  # Skip incomplete blocks at the end
        topk_indices = torch.topk(block.abs(), n, sorted=False)[1]
        mask.view(-1)[i:i + m].scatter_(0, topk_indices, 1)

    sparse_param = param * mask
    return sparse_param, mask


def block_random_pruning(state_dict1, state_dict2, sparsity_level=0.5, pruning_method='magnitude', n=None, m=None):
    """Performs block random pruning on two state dicts."""
    pruned_state_dict1 = {}
    pruned_state_dict2 = {}

    for name in state_dict1.keys():
        if name in state_dict2:
            param1 = state_dict1[name]
            param2 = state_dict2[name]

            if pruning_method == 'magnitude':
                param1, mask1 = topk_values_mask(param1, sparsity_level=sparsity_level, return_mask=True)
                param2 *= (1 - mask1)
                param2, mask2 = topk_values_mask(param2, sparsity_level=sparsity_level, return_mask=True)
            elif pruning_method == 'random':
                mask1 = torch.bernoulli(torch.full(param1.shape, 1 - sparsity_level)).float()
                param1 *= mask1
                param2 *= (1 - mask1)
                mask2 = torch.bernoulli(torch.full(param2.shape, 1 - sparsity_level)).float()
                param2 *= mask2
            elif pruning_method == 'n:m' and n is not None and m is not None:
                param1, mask1 = n_m_block_pruning(param1, n=n, m=m)
                param2 *= (1 - mask1)
                param2, mask2 = n_m_block_pruning(param2, n=n, m=m)
            else:
                raise ValueError(f"Unsupported pruning method: {pruning_method}")

            pruned_state_dict1[name] = param1
            pruned_state_dict2[name] = param2

    return pruned_state_dict1, pruned_state_dict2


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Perform block pruning on task vectors.")
    parser.add_argument('--model1_path', type=str, required=True, help="Path to model 1.")
    parser.add_argument('--model2_path', type=str, required=True, help="Path to model 2.")
    parser.add_argument('--save_path1', type=str, required=True, help="Path to save pruned model 1.")
    parser.add_argument('--save_path2', type=str, required=True, help="Path to save pruned model 2.")
    parser.add_argument('--sparsity_level', type=float, default=0.5, help="Target sparsity level.")
    parser.add_argument('--pruning_method', type=str, choices=['magnitude', 'random', 'n:m'], default='magnitude', help="Pruning method to use.")
    parser.add_argument('--n', type=int, default=None, help="Value of n for n:m pruning.")
    parser.add_argument('--m', type=int, default=None, help="Value of m for n:m pruning.")
    args = parser.parse_args()

    # Load models
    state_dict1 = torch.load(args.model1_path, map_location='cpu')
    state_dict2 = torch.load(args.model2_path, map_location='cpu')

    # Perform block pruning
    pruned_state_dict1, pruned_state_dict2 = block_random_pruning(
        state_dict1, state_dict2, sparsity_level=args.sparsity_level, pruning_method=args.pruning_method, n=args.n, m=args.m
    )

    # Save pruned models
    torch.save(pruned_state_dict1, args.save_path1)
    torch.save(pruned_state_dict2, args.save_path2)

    print("Pruned models have been saved.")


# Execute main function
if __name__ == "__main__":
    main()
