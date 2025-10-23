import torch


class NoiseScheule:
    def __init__(self, n_steps, min_val=1e-3, max_val=1.):
        self.min_val = min_val
        self.max_val = max_val
        self.n_steps = n_steps
        self.values = torch.linspace(min_val, max_val, n_steps)
    
    def __call__(self, x):
        steps = torch.rand(self.n_steps, (x.size(0),))
        return self.values[steps].to(x)

    def sample(self, step):
