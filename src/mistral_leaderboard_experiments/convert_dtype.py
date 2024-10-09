import torch
from transformers import AutoModelForCausalLM

def convert_model_dtype(model_path, save_path):
    # Load pretrained model
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Convert model parameters to float16
    for name, param in model.named_parameters():
        param.data = param.data.to(dtype=torch.float16)

    # Save the model with float16 precision
    model.save_pretrained(save_path)

    print(f"Model converted to float16 and saved at: {save_path}")

if __name__ == "__main__":
    pretrained_model_path = "path/to/Mistral-7b-v0.1/"
    save_path = "/path/to/save/Mistral-7b-v0.1-float16/"

    convert_model_dtype(pretrained_model_path, save_path)
