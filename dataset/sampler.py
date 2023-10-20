import torch


class DataSampler(object):
    """Samples training points."""
    def __init__(self, nr_of_points=1000, p_surface=0.3, p_offset=0.3, p_grid=0.4):
        assert p_surface + p_offset + p_grid == 1
        self.nr_of_points = nr_of_points
        self.p_surface = p_surface
        self.p_offset = p_offset
        self.p_grid = p_grid

    def __call__(self, data):
        surface_points = int(self.nr_of_points * self.p_surface)
        offset_points = int(self.nr_of_points * self.p_offset)
        grid_points = int(self.nr_of_points * self.p_grid)

        perm = torch.randperm(data["surface_points"].shape[0])
        idx = perm[:surface_points]
        surface_samples = data["surface_points"][idx]
        surface_samples_sdf = data["surface_sdf"][idx]

        perm = torch.randperm(data["offset_points"].shape[0])
        idx = perm[:offset_points]
        offset_samples = data["offset_points"][idx]
        offset_samples_sdf = data["offset_sdf"][idx]

        perm = torch.randperm(data["grid_points"].shape[0])
        idx = perm[:grid_points]
        grid_samples = data["grid_points"][idx]
        grid_samples_sdf = data["grid_sdf"][idx]

        coords = torch.cat((surface_samples, offset_samples, grid_samples), dim=0).float()
        sdf = torch.cat((surface_samples_sdf, offset_samples_sdf, grid_samples_sdf), dim=0).float()

        sampled_data = {
            "surface_points": data["surface_points"],
            "surface_normals": data["surface_normals"],
            "sampled_points": coords,
            "sampled_sdf": sdf
        }

        return sampled_data
