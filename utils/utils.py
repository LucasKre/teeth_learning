import mcubes
import torch
import trimesh
from tqdm import tqdm
from dataset.preprocessing import Compose, MoveMeshToCenter, NormalizeMesh, MeshToSdf


def create_cube(resolution):
    """Create a cube with the given resolution."""
    overall_index = torch.arange(0, resolution ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(resolution ** 3, 4)

    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (resolution - 1)

    samples[:, 2] = overall_index % resolution
    samples[:, 1] = (overall_index.long().float() / resolution) % resolution
    samples[:, 0] = ((overall_index.long().float() / resolution) / resolution) % resolution

    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    samples.requires_grad = False

    return samples


def sdf_to_mesh(cube_sdf, level_set=0.0, smooth=True):
    """Convert a cube to a mesh."""
    numpy_3d_sdf_tensor = cube_sdf.numpy()

    voxel_size = 2.0 / (cube_sdf.shape[0] - 1)

    if smooth:
        numpy_3d_sdf_tensor = mcubes.smooth(numpy_3d_sdf_tensor)
    verts, faces = mcubes.marching_cubes(numpy_3d_sdf_tensor, level_set)

    mesh = trimesh.Trimesh(verts, faces)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_inversion(mesh)
    return mesh


def reconstruct_mesh(mesh, network, n=128, max_batch=20000, smooth=True):
    """Reconstruct a mesh by using a trained network."""
    device = network.device
    transform = Compose(
        [MoveMeshToCenter(),
         NormalizeMesh(),
         MeshToSdf(grid_min=-1, grid_max=1)]
    )
    data = transform(mesh)
    pc = data["surface_points"].to(device).float()
    pc_n = data["surface_normals"].to(device).float()
    cube = create_cube(n)
    cube_points = cube.shape[0]
    head = 0
    for i in tqdm(range(0, cube_points, max_batch)):
        # while head < cube_points:
        query = cube[head: min(head + max_batch, cube_points), 0:3].to(device).unsqueeze(0)

        pred_sdf = network.network.predict_sdf(pc, pc_n, query)

        cube[head: min(head + max_batch, cube_points), 3] = pred_sdf.squeeze()

        head += max_batch
    sdf_values = cube[:, 3]
    sdf_values = sdf_values.reshape(n, n, n)
    mesh_rec = sdf_to_mesh(sdf_values, smooth=smooth)
    return mesh_rec


def reconstruction_error_mesh(mesh, network):
    """Calculate the reconstruction error for a mesh."""
    device = network.device
    transform = Compose(
        [MoveMeshToCenter(),
         NormalizeMesh()]
    )
    mesh_to_sdf = MeshToSdf()
    mesh_tranformed = transform(mesh)
    data = mesh_to_sdf(mesh_tranformed)
    query = mesh_tranformed.vertices[mesh_tranformed.faces].mean(axis=1)
    query = torch.from_numpy(query).unsqueeze(0).to(network.device).float()
    pc = data["surface_points"].to(device).float()
    pc_n = data["surface_normals"].to(device).float()

    pred_sdf = network.network.predict_sdf(pc, pc_n, query).flatten()

    error = torch.abs(0 - pred_sdf)
    # error = error.max() - error
    return error + 0.0000001


def rgb(minimum, maximum, value):
    """Convert a value to a rgb color."""
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value - minimum) / (maximum - minimum)
    b = int(max(0, 255 * (1 - ratio)))
    r = int(max(0, 255 * (ratio - 1)))
    g = 255 - b - r
    return r, g, b


def distance_to_rgb(distances):
    """Convert a list of distances to a list of rgb colors."""
    rgb_list = []

    min_distance = min(distances)
    max_distance = max(distances)

    for distance in distances:
        red, green, blue = rgb(min_distance, max_distance, distance)
        rgb_list.append([red, green, blue, 255])

    return rgb_list


def color_reconstruction_error(mesh, network):
    """Color a mesh by its reconstruction error."""
    error = reconstruction_error_mesh(mesh, network)
    mesh.visual.vertex_colors = distance_to_rgb(error.detach().cpu().numpy())
    return mesh
