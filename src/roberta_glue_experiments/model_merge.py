import torch

def add_model_params(model1_path, model2_path, model3_path, save_path, lamb1, lamb2):
    model1 = torch.load(model1_path, map_location='cuda:0')
    model2 = torch.load(model2_path, map_location='cuda:0')
    model3 = torch.load(model3_path, map_location='cuda:0')

    for key in model1.keys():
        if key in model2.keys():
            model2[key] = model1[key] * lamb1 + model2[key] * lamb2
        if key in model3.keys():
            model2[key] = model3[key] + model2[key]

    torch.save(model2, save_path)
