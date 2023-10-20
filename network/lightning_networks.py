import torch
import lightning.pytorch as pl
from network.sdf_encoder import SDFEncoder




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
        x, sdf = batch["sampled_points"], batch["sampled_sdf"]
        pred_sdf = self.network(pc, pc_n, x)
        loss = self.criterion(pred_sdf, sdf)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True, on_step=False, batch_size=self.config["training"]["batch_size"])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config["training"]["lr"])
        return optimizer


