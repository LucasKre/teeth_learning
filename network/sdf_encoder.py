import torch
from torch import nn

from network.mesh_encoder import DGCNNEncoder
from network.conv_pointnet_encoder import ConvPointnet
from network.sdf_network import SDFNetwork
import json


class SDFEncoder(nn.Module):
    def __init__(self, config):
        super(SDFEncoder, self).__init__()
        self.encoder_type = config["network"]["mesh_encoder"]["encoder_type"]
        if self.encoder_type == "DGCNN":
            self.mesh_encoder = DGCNNEncoder(
                nr_of_points=config["network"]["mesh_encoder"]["nr_of_points"],
                k=config["network"]["mesh_encoder"]["k"],
                hidden_dims=config["network"]["mesh_encoder"]["hidden_dims"],
                encod_dim=config["network"]["mesh_encoder"]["encod_dim"],
                dropout_prob=config["network"]["mesh_encoder"]["dropout"],
                include_normals=config["network"]["mesh_encoder"]["include_normals"],
            )
        elif self.encoder_type == "ConvPointNet":
            self.mesh_encoder = ConvPointnet(c_dim=config["network"]["mesh_encoder"]["encod_dim"],
                                             hidden_dim=config["network"]["mesh_encoder"]["hidden_dim"],
                                             plane_resolution=config["network"]["mesh_encoder"]["plane_resolution"],
                                             n_blocks=config["network"]["mesh_encoder"]["nr_of_blocks"],)
        else:
            raise NotImplementedError
        self.sdf_network = SDFNetwork(
            cond_dim=config["network"]["mesh_encoder"]["encod_dim"],
            hidden_dims=config["network"]["sdf_decoder"]["hidden_dims"],
        )

    def forward(self, pc, pc_n, centroid, x):
        batch_size = x.size(0)
        if self.encoder_type == "DGCNN":
            cond = self.mesh_encoder(pc, pc_n)
            cond = cond.repeat(batch_size, 1)
        elif self.encoder_type == "ConvPointNet":
            cond = self.mesh_encoder(pc, x, centroid)
        sdf = self.sdf_network(x, cond)
        return sdf

    def predict_sdf(self, pc, pc_n, centroid, coords):
        with torch.no_grad():
            pc = pc.unsqueeze(0)
            centroid = centroid.unsqueeze(0)
            return self.forward(pc, pc_n, centroid, coords)



