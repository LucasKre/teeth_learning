import numpy as np
import torch
import trimesh
from pysdf import SDF
from scipy.spatial import KDTree


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


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

    def __init__(self, surface_points=60000,
                 offset_points=60000,
                 offset_s=0.1,
                 grid_resolution=42,
                 grid_min=-1,
                 grid_max=1,
                 include_normals=True):
        self.surface_points = surface_points
        self.offset_points = offset_points
        self.offset_s = offset_s
        self.grid_resolution = grid_resolution
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.include_normals = include_normals

    def __call__(self, mesh):
        # transform to sdf
        sdf_f = SDF(mesh.vertices, mesh.faces)
        # sample surface points
        random_surface_points = sdf_f.sample_surface(self.surface_points)
        surface_sdf = np.zeros(self.surface_points)

        # surface normals
        surface_normals = None
        if self.include_normals:
            # CAUTION: this only estimates the surface normals based on the nearest face
            kd_tree = KDTree(mesh.vertices[mesh.faces].mean(axis=1))
            _, index = kd_tree.query(random_surface_points)
            surface_normals = mesh.face_normals[index]

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

        out_dict = {
            "surface_points": torch.from_numpy(random_surface_points),
            "surface_normals": torch.from_numpy(surface_normals) if surface_normals is not None else None,
            "surface_sdf": torch.from_numpy(surface_sdf),
            "offset_points": torch.from_numpy(random_offset_points),
            "offset_sdf": torch.from_numpy(offset_sdf),
            "grid_points": torch.from_numpy(grid_coords),
            "grid_sdf": torch.from_numpy(grid_sdf)
        }

        return out_dict


class MeshToLocalSubParts:
    """Convert a mesh to several local sub parts."""

    def __init__(self, nr_of_locals=5, distance=0.3):
        self.nr_of_locals = nr_of_locals
        self.distance = distance
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __call__(self, mesh):
        pc = torch.from_numpy(mesh.vertices[mesh.faces].mean(axis=1)).to(self.device).float()
        centroids = farthest_point_sample(pc.unsqueeze(0), self.nr_of_locals)
        centroids = centroids.squeeze(0)
        sub_meshes = []
        for c_idx in centroids:
            c = pc[c_idx]
            dist = torch.norm(pc - c, dim=1)
            mask = dist < self.distance
            mask = mask.cpu().numpy()
            sub_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[mask])
            sub_meshes.append(sub_mesh)
        centroid_coords = pc[centroids].cpu()
        return sub_meshes, centroid_coords


class LocalMeshToSdf:

    def __init__(self, transform=MeshToSdf()):
        self.transform = transform

    def __call__(self, data):
        local_meshes, centroids = data
        for mesh, c in zip(local_meshes, centroids):
            sdf_dift = self.transform(mesh)
            sdf_dift["centroid"] = c
            yield sdf_dift
