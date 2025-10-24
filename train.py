import os, time
import hydra, wandb
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MovingMNIST

from model import DiT
from noise import LinearSchedule, CumRetentionSchedule
from utils import gen_ids, make_tuple


def train(config):
    dataset = MovingMNIST(root="data", download=True)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
    )

    model = torch.compile(DiT(config.model).cuda())
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()
    diffuser = CumRetentionSchedule(1000)
    ph, pw = make_tuple(config.patch_size)
    h, w = make_tuple(config.model.frame_size)
    x_loc = gen_ids(config.batch_size, config.frames, h//ph, w//pw)
    x_loc = x_loc.cuda()

    for epoch in range(1, config.epochs+1):
        pb = tqdm(dataloader, desc=f"{epoch:>2}/{config.epochs}")
        for x_0 in pb:
            x_0 = x_0.cuda() / 255. 

            x_t, timesteps = diffuser(x_0)
            denoised_x = model(x_t, x_loc, timesteps)
            
            loss = criterion(denoised_x, x_0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({"loss": loss.item()})
            pb.set_postfix(loss=loss.item())
        
        if epoch % config.checkpoint_every == 0:
            checkpoint_path = f"video_dit_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)



@hydra.main(config_path=".", config_name="config.yaml", version_base=None)
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