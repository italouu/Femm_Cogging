import torch
from functools import reduce
import operator

def l2_loss(out, y):
    l2 = torch.mean((out - y) ** 2)
    return l2

def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c