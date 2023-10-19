import torch

from torch import nn
from network.network_utils import ResNetBlock


class SDFNetwork(nn.Module):
    """SDF Network"""

    def __init__(self, cond_dim=128, hidden_dims=[128, 256, 512, 1024]):
        super(SDFNetwork, self).__init__()
        self.cond_dim = cond_dim
        self.coords_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.block = nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i == 0:
                self.block.append(ResNetBlock(64 + cond_dim, hidden_dims[i]))
            else:
                self.block.append(ResNetBlock(hidden_dims[i - 1], hidden_dims[i]))
        self.out = nn.Sequential(
            nn.Linear(hidden_dims[-1], 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, coords, cond):
        coords = self.coords_encoder(coords)
        x = torch.cat((coords, cond), dim=1)
        for i in range(len(self.block)):
            x = self.block[i](x)
        x = self.out(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    coors = torch.rand(32, 3).to(device)
    cond = torch.rand(32, 128).to(device)
    model = SDFNetwork().to(device)
    out = model(coors, cond)
    print(out.shape)
