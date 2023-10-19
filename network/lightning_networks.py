import torch
import lightning.pytorch as pl
from network.sdf_encoder import SDFEncoder


def sample_from_data(batch, nr_of_points=1000, p_surface=0.3, p_offset=0.3, p_grid=0.4):
    assert p_surface + p_offset + p_grid == 1

    surface_points = int(nr_of_points * p_surface)
    offset_points = int(nr_of_points * p_offset)
    grid_points = int(nr_of_points * p_grid)

    perm = torch.randperm(batch["surface_points"].shape[0])
    idx = perm[:surface_points]
    surface_samples = batch["surface_points"][idx]
    surface_samples_sdf = batch["surface_sdf"][idx]

    perm = torch.randperm(batch["offset_points"].shape[0])
    idx = perm[:offset_points]
    offset_samples = batch["offset_points"][idx]
    offset_samples_sdf = batch["offset_sdf"][idx]

    perm = torch.randperm(batch["grid_points"].shape[0])
    idx = perm[:grid_points]
    grid_samples = batch["grid_points"][idx]
    grid_samples_sdf = batch["grid_sdf"][idx]

    coords = torch.cat((surface_samples, offset_samples, grid_samples), dim=0).float()
    sdf = torch.cat((surface_samples_sdf, offset_samples_sdf, grid_samples_sdf), dim=0).float()
    return coords, sdf


class LitSDFEncoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.network = SDFEncoder(config)
        self.criterion = torch.nn.L1Loss()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        pc = batch["surface_points"].float()
        pc_n = batch["surface_normals"].float()
        pc = pc.unsqueeze(0)
        pc_n = pc_n.unsqueeze(0)
        x, sdf = sample_from_data(batch, nr_of_points=self.config["training"]["batch_size"])
        pred_sdf = self.network(pc, pc_n, x)
        loss = self.criterion(pred_sdf, sdf)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True, on_step=False, batch_size=self.config["training"]["batch_size"])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config["training"]["lr"])
        return optimizer


