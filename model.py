import torch
from torch import nn
import torch.nn.functional as F

import math
from dataclasses import dataclass

def attention(q, k, v, pos):
    ...


def rope(pos, dim, theta):
    ...


def apply_rope(q, k, freqs):
    ...

def make_tuple(num_or_tuple):
    if isinstance(num_or_tuple, tuple):
        return num_or_tuple
    return (num_or_tuple, num_or_tuple)


class Modulation(nn.Module):
    ...


class Streamblock(nn.Module):
    ...    


class DiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.blocks = ...
        self.time_ = ...