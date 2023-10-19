import mcubes
import torch
import trimesh


def create_cube(resolution):
    """Create a cube with the given resolution."""
    overall_index = torch.arange(0, resolution ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(resolution ** 3, 4)

    # the voxel_origin is the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (resolution - 1)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % resolution
    samples[:, 1] = (overall_index.long().float() / resolution) % resolution
    samples[:, 0] = ((overall_index.long().float() / resolution) / resolution) % resolution

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    samples.requires_grad = False

    return samples


def sdf_to_mesh(cube_sdf, level_set=0.0, smooth=True):
    """Convert a cube to a mesh."""
    numpy_3d_sdf_tensor = cube_sdf.numpy()

    if smooth:
        numpy_3d_sdf_tensor = mcubes.smooth(numpy_3d_sdf_tensor)
    verts, faces = mcubes.marching_cubes(numpy_3d_sdf_tensor, level_set)

    mesh = trimesh.Trimesh(verts, faces)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_inversion(mesh)
    return mesh


