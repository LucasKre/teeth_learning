import os

import torch
import trimesh
from torch.utils.data import Dataset
from tqdm import tqdm


class BaseDataset(Dataset):
    def __init__(self, root_dir, mesh_dir, process_dir, preprocessing=None):
        super(BaseDataset, self).__init__()
        self.root_dir = root_dir
        self.mesh_dir = mesh_dir
        self.process_dir = process_dir
        self.preprocessing = preprocessing
        # create folder if not existing
        if not os.path.exists(os.path.join(self.root_dir, self.process_dir)):
            os.makedirs(os.path.join(self.root_dir, self.process_dir))
        if not self.__is_preprocessed():
            self.__preprocess()
        self.file_names = os.listdir(
            os.path.join(self.root_dir, self.process_dir)
        )

    def __is_preprocessed(self):
        # list all files in mesh_dir
        meshes = os.listdir(
            os.path.join(self.root_dir, self.mesh_dir)
        )
        # list all files in process_dir
        processed = os.listdir(
            os.path.join(self.root_dir, self.process_dir)
        )
        return len(meshes) == len(processed)

    def __preprocess(self):
        assert self.preprocessing is not None
        meshes = os.listdir(
            os.path.join(self.root_dir, self.mesh_dir)
        )
        print(f"Preprocessing... Saving to {self.process_dir}")
        for m in tqdm(meshes):
            mesh = trimesh.load(os.path.join(self.root_dir, self.mesh_dir, m))
            data = self.preprocessing(mesh)
            file_name = f"{m.split('.')[0]}.pt"
            # save to process_dir
            torch.save(data, os.path.join(self.root_dir, self.process_dir, file_name))

    def __getitem__(self, index):
        data = torch.load(
            os.path.join(self.root_dir, self.process_dir, self.file_names[index])
        )
        return data

    def __len__(self):
        return len(self.file_names)


