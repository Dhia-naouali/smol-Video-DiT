import os, time
import hydra, wandb
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MovingMNIST

from .model import DiT

torch.set_default_device("cuda:0")

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

    for epoch in range(config.epochs):
        pb = tqdm(dataloader, desc=f"{epoch:>2}/{config.epochs}")
        for ... in pb:
            ...




@hydra.main(config_path=".", config_name="defaults.yaml")
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