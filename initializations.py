import torch
import math
from torch.nn.init import _calculate_fan_in_and_fan_out

def init_normal(module, mean=0.0, std=0.1):

    if hasattr(module, 'weight') and module.weight is not None:
        with torch.no_grad():
            module.weight.normal_(mean, std)

    if hasattr(module, 'bias') and module.bias is not None:
        with torch.no_grad():
            module.bias.zero_()
    return module

def init_xavier_normal(module, gain=1.0):

    if hasattr(module, 'weight') and module.weight is not None:
        fan_in, fan_out = _calculate_fan_in_and_fan_out(module.weight)
        std = gain * math.sqrt(2.0 / (fan_in + fan_out)) # https://www.geeksforgeeks.org/xavier-initialization/
        with torch.no_grad():
            module.weight.normal_(0, std)

    if hasattr(module, 'bias') and module.bias is not None:
        with torch.no_grad():
            module.bias.zero_()
    return module

def he_normal(module, a=0, mode='fan_in'):

    if hasattr(module, 'weight') and module.weight is not None:
        fan_in, _ = _calculate_fan_in_and_fan_out(module.weight)
        gain = math.sqrt(2.0)            
        std = gain / math.sqrt(fan_in) #https://www.geeksforgeeks.org/kaiming-initialization-in-deep-learning/
        with torch.no_grad():
            module.weight.normal_(0, std)

    if hasattr(module, 'bias') and module.bias is not None:
        with torch.no_grad():
            module.bias.zero_()
    return module

def init_lecun_normal(module):

    if hasattr(module, 'weight') and module.weight is not None:
        fan_in, _ = _calculate_fan_in_and_fan_out(module.weight)
        std = math.sqrt(1.0 / fan_in) #https://www.geeksforgeeks.org/lecun-initialization-in-deep-learning/
        with torch.no_grad():
            module.weight.normal_(0, std)

    if hasattr(module, 'bias') and module.bias is not None:
        with torch.no_grad():
            module.bias.zero_()
    return module

def init_zeros(module):
    
    if hasattr(module, 'weight') and module.weight is not None:
        with torch.no_grad():
            module.weight.zero_()

    if hasattr(module, 'bias') and module.bias is not None:
        with torch.no_grad():
            module.bias.zero_()
    return module

def initialize_model(model, init_method='normal', **kwargs):
    init_functions = {
        'normal': init_normal,
        'xavier_normal': init_xavier_normal,
        'he_normal': he_normal,    
        'lecun_normal': init_lecun_normal,
        'zeros': init_zeros,
    }
    init_function = init_functions[init_method]
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            init_function(module, **kwargs)
    return model
