import torch



def make_tuple(num_or_tuple):
    if isinstance(num_or_tuple, tuple):
        return num_or_tuple
    return (num_or_tuple, num_or_tuple)


def gen_ids(B, T, nh, nw):
    L = T * nh * nw
    tt, yy, xx = torch.meshgrid(
        torch.arange(T), torch.arange(nh), torch.arange(nw), indexing="ij"
    )
    return torch.stack([tt, yy, xx], dim=-1).reshape(1, L, 3).expand(B, -1, -1).to(torch.float32)
