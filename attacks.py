import torch

def simple_poison(model):
    """Adds random noise to model parameters to simulate a basic adversarial attack."""
    for param in model.parameters():
        param.data.add_(torch.randn_like(param) * 0.2)
    return model
