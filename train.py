import os, time
import hydra, wandb
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MovingMNIST

from .model import DiT
from .noise import NoiseScheule

torch.set_default_device("cuda:0")


def gen_ids(B, T, nh, nw):
    L = T * nh * nw
    tt, yy, xx = torch.meshgrid(
        torch.arange(T), torch.arange(nh), torch.arange(nw), indexing="ij"
    )
    return torch.stack([tt, yy, xx], dim=-1).reshape(1, L, 3).expand(B, -1, -1).to(torch.float32)


def train(config):
    dataset = MovingMNIST(root="data", download=True)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
    )

    model = torch.compile(DiT(config.model))
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()
    noise_scheduler = NoiseScheule(1e3)
    x_loc = gen_ids(config.batch_size, config.frames, config.pw, config.ph)

    for epoch in range(1, config.epochs+1):
        pb = tqdm(dataloader, desc=f"{epoch:>2}/{config.epochs}")
        for x in pb:
            x = x / 255.

            timesteps = noise_scheduler(x).view(-1, [1]*(x.ndim-1))
            noise = torch.rand_like(x)
            noisy_x = x + timesteps * noise
            denoised_x = model(noisy_x, x_loc, timesteps)
            
            loss = criterion(denoised_x, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log(loss=loss.itme())
            pb.set_postfix(loss=loss.item())
        
        if epoch % config.checkpoint_every == 0:
            checkpoint_path = f"video_dit_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)



@hydra.main(config_path=".", config_name="config.yaml")
def main(config):
    wandb.init(
        project="Video-Dit",
        name=f"run_{int(time.time())}",
        config=OmegaConf.to_container(config, resolve=True)
    )
    train(config)
    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()