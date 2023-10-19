import torch
from torch import nn
import torch.nn.functional as F


def knn(x, k):
    x_t = x.transpose(2, 1)
    pairwise_distance = torch.cdist(x_t, x_t, p=2)
    idx = pairwise_distance.topk(k=k + 1, dim=-1, largest=False)[1][:, :, 1:]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx_org=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    if idx_org is None:
        idx_org = knn(x, k=k)

    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx_org + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature, idx_org  # (batch_size, 2*num_dims, num_points, k)


class DGCNNEncoder(nn.Module):
    def __init__(self, nr_of_points, k=32, hidden_dims=[32, 64, 128, 256], encod_dim=128, dropout_prob=0.2, include_normals=False):
        super(DGCNNEncoder, self).__init__()
        self.nr_of_points = nr_of_points
        self.k = k
        self.encod_dim = encod_dim
        self.include_normals = include_normals
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i == 0:
                self.layers.append(nn.Sequential(
                    nn.Conv2d(6, hidden_dims[i], kernel_size=1, bias=False),
                    nn.InstanceNorm2d(hidden_dims[i]),
                    nn.LeakyReLU(negative_slope=0.2)
                ))
            else:
                self.layers.append(nn.Sequential(
                    nn.Conv2d(hidden_dims[i - 1] * 2, hidden_dims[i], kernel_size=1, bias=False),
                    nn.InstanceNorm2d(hidden_dims[i]),
                    nn.LeakyReLU(negative_slope=0.2)
                ))
        if include_normals:
            self.layers_n = nn.ModuleList()
            for i in range(len(hidden_dims)):
                if i == 0:
                    self.layers_n.append(nn.Sequential(
                        nn.Conv2d(6, hidden_dims[i], kernel_size=1, bias=False),
                        nn.InstanceNorm2d(hidden_dims[i]),
                        nn.LeakyReLU(negative_slope=0.2)
                    ))
                else:
                    self.layers_n.append(nn.Sequential(
                        nn.Conv2d(hidden_dims[i - 1] * 2, hidden_dims[i], kernel_size=1, bias=False),
                        nn.InstanceNorm2d(hidden_dims[i]),
                        nn.LeakyReLU(negative_slope=0.2)
                    ))
        in_hidden = sum(hidden_dims) * 2 if include_normals else sum(hidden_dims)
        self.hidden = nn.Sequential(
            nn.Conv1d(in_hidden, encod_dim, kernel_size=1, bias=False),
            nn.InstanceNorm1d(encod_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout_prob))

        self.fc = nn.Sequential(
            nn.Linear(encod_dim, encod_dim),
            nn.InstanceNorm1d(encod_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(encod_dim, encod_dim)
        )

    def forward(self, x, n):
        batch_size = x.size(0)
        x = x.transpose(2, 1)
        n = n.transpose(2, 1)
        res = []
        res_n = []
        for i in range(len(self.layers)):
            x, idx = get_graph_feature(x, k=self.k)
            x = self.layers[i](x)
            x = x.max(dim=-1, keepdim=False)[0]
            res.append(x)
            if self.include_normals:
                n, idx = get_graph_feature(n, k=self.k, idx_org=idx)
                n = self.layers_n[i](n)
                n = n.max(dim=-1, keepdim=False)[0]
                res_n.append(n)
        res = res + res_n
        x = torch.cat(res, dim=1)
        x = self.hidden(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.rand(2, 2048, 3).to(device)
    n = torch.rand(2, 2048, 3).to(device)
    model = DGCNNEncoder(2048, include_normals=True).to(device)
    out = model(x, n)
    print(out.shape)
