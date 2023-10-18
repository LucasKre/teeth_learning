import numpy as np
from pysdf import SDF


class Compose:
    """Compose several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class MoveMeshToCenter:
    """Move the mesh to the origin."""

    def __call__(self, mesh):
        mesh.vertices -= mesh.vertices.mean(axis=0)
        return mesh


class NormalizeMesh:
    """Normalize the mesh to fit within a unit sphere centered at the origin."""

    def __call__(self, mesh):
        # Get the vertices of the mesh
        vertices = np.array(mesh.vertices)

        # Calculate the maximum distance from the origin (mesh center)
        center = np.mean(vertices, axis=0)
        distances = np.linalg.norm(vertices - center, axis=1)
        max_distance = np.max(distances)

        # Calculate the scaling factor to fit within a unit sphere
        scale_factor = 1.0 / max_distance

        # Scale the mesh
        scaled_mesh = mesh.copy()
        scaled_mesh.apply_scale(scale_factor)
        return scaled_mesh


class MeshToSdf:
    """Convert a mesh to a signed distance field."""

    def __init__(self, surface_points=10000,
                 offset_points=10000,
                 offset_s=0.1,
                 grid_resolution=32,
                 grid_min=-1,
                 grid_max=1):
        self.surface_points = surface_points
        self.offset_points = offset_points
        self.offset_s = offset_s
        self.grid_resolution = grid_resolution
        self.grid_min = grid_min
        self.grid_max = grid_max

    def __call__(self, mesh):
        # transform to sdf
        sdf_f = SDF(mesh.vertices, mesh.faces)
        # sample surface points
        random_surface_points = sdf_f.sample_surface(self.surface_points)
        surface_sdf = np.zeros(self.surface_points)

        # sample offset points
        random_offset = np.random.normal(0, self.offset_s, (self.offset_points, 3))
        random_offset_points = random_surface_points + random_offset
        offset_sdf = sdf_f(random_offset_points)

        # sample grid points
        range_min, range_max = self.grid_min, self.grid_max
        x_range = (range_min, range_max)
        y_range = (range_min, range_max)
        z_range = (range_min, range_max)
        grid_coords = np.array(
            np.meshgrid(*[np.linspace(*r, self.grid_resolution) for r in [x_range, y_range, z_range]])).T.reshape(-1, 3)
        # sample in grid
        grid_sdf = sdf_f(grid_coords)

        return (random_surface_points, surface_sdf), (random_offset_points, offset_sdf), (grid_coords, grid_sdf)
