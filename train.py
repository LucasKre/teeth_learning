import argparse
import json

import torch
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader
import random
import numpy as np
from lightning.pytorch import seed_everything
from dataset.dataset import BaseDataset
import lightning.pytorch as pl

from dataset.preprocessing import Compose, MoveMeshToCenter, NormalizeMesh, MeshToSdf
from network.sdf_encoder import SDFEncoder
from network.lightning_networks import LitSDFEncoder
# set seed
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(SEED)

torch.set_float32_matmul_precision('medium')

random.seed(SEED)

seed_everything(SEED, workers=True)


def train(config):
    #set up dataset and network
    transform = Compose(
        [MoveMeshToCenter(),
         NormalizeMesh(),
         MeshToSdf(grid_min=-1, grid_max=1)]
    )
    dataset = BaseDataset(root_dir=config["dataset"]["root_dir"],
                          mesh_dir=config["dataset"]["mesh_dir"],
                          process_dir=config["dataset"]["process_dir"],
                          preprocessing=transform,
                          in_memory=config["dataset"]["in_memory"])
    lit_network = LitSDFEncoder(config)

    #compile model
    # lit_network.network = torch.compile(lit_network.network)

    logger = TensorBoardLogger(save_dir=config["training"]["log_dir"])

    checkpoint_callback = ModelCheckpoint(dirpath=logger.log_dir, save_top_k=1, monitor="train_loss", mode="min", save_last=False)

    #set up pl trainer
    trainer = pl.Trainer(max_epochs=config["training"]["epochs"], accelerator='cuda',
                         enable_progress_bar=True,  precision=config["training"]["precision"], deterministic=False,
                         logger=logger, callbacks=[checkpoint_callback, ModelSummary(max_depth=2)],
                         devices=config["training"]["devices"])

    #train
    trainer.fit(lit_network, dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3D deep learning experiments")
    parser.add_argument("--config", type=str, default="config/config.json", help="Path to config file")
    args = parser.parse_args()

    config = args.config
    config = json.load(open(config))

    train(config)



