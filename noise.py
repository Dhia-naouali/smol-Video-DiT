import torch


class LinearSchedule:
    def __init__(self, n_steps, min_val=1e-3, max_val=1.):
        self.min_val = min_val
        self.max_val = max_val
        self.n_steps = n_steps
        self.values = torch.linspace(min_val, max_val, n_steps)
    
    def __call__(self, x):
        noise = torch.randn_like(x)
        
        steps = torch.randint(self.n_steps, (x.size(0),))
        steps = self.values[steps].view(-1, *([1] * (x.ndim-1)))
        x_t = x + steps * noise
        return x_t, steps


class CumRetentionSchedule:
    def __init__(self, n_steps, beta_start=1e-4, beta_end=.02):
        self.values = torch.cumprod(
            1. - torch.linspace(beta_start, beta_end, n_steps)
        )

    def __call__(self, x):
        noise = torch.randn_like(x)        
        steps = torch.randint(0, self.n_steps, (x.size(0),))
        steps = self.values[steps].view(-1, *([1] * (x.ndim-1)))

        x_t = steps.sqrt() * x + (1 - steps).sqrt() * noise
        return x_t, steps