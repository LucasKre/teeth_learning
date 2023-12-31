import os

import torch
import trimesh
from torch.utils.data import Dataset
from tqdm import tqdm
import types

class BaseDataset(Dataset):
    def __init__(self, root_dir, mesh_dir, process_dir, preprocessing=None, in_memory=False, sampler=None):
        super(BaseDataset, self).__init__()
        self.root_dir = root_dir
        self.mesh_dir = mesh_dir
        self.process_dir = process_dir
        self.preprocessing = preprocessing
        self.in_memory = in_memory
        self.sampler = sampler
        # create folder if not existing
        if not os.path.exists(os.path.join(self.root_dir, self.process_dir)):
            os.makedirs(os.path.join(self.root_dir, self.process_dir))
        if not self.__is_preprocessed():
            self.__preprocess()
        self.file_names = os.listdir(
            os.path.join(self.root_dir, self.process_dir)
        )
        if self.in_memory:
            print("Loading data into memory...")
            self.data = []
            for i in tqdm(range(len(self.file_names))):
                self.data.append(self.__load_from_file(i))

    def __is_preprocessed(self):
        # list all files in mesh_dir
        meshes = os.listdir(
            os.path.join(self.root_dir, self.mesh_dir)
        )
        # list all files in process_dir
        processed = os.listdir(
            os.path.join(self.root_dir, self.process_dir)
        )
        return len(processed) >= len(meshes)

    def __preprocess(self):
        assert self.preprocessing is not None
        meshes = os.listdir(
            os.path.join(self.root_dir, self.mesh_dir)
        )
        print(f"Preprocessing... Saving to {self.process_dir}")
        for m in tqdm(meshes):
            mesh = trimesh.load(os.path.join(self.root_dir, self.mesh_dir, m))
            data = self.preprocessing(mesh)
            # check if data is a generator -> can happen if the preprocessing step yield results
            if isinstance(data, types.GeneratorType):
                for i, d in enumerate(data):
                    file_name = f"{m.split('.')[0]}_{i}.pt"
                    torch.save(d, os.path.join(self.root_dir, self.process_dir, file_name))
            else:
                file_name = f"{m.split('.')[0]}.pt"
                torch.save(data, os.path.join(self.root_dir, self.process_dir, file_name))

    def __load_from_file(self, index):
        return torch.load(os.path.join(self.root_dir, self.process_dir, self.file_names[index]))

    def __getitem__(self, index):
        if self.in_memory:
            data = self.data[index]
        else:
            data = self.__load_from_file(index)
        if self.sampler is not None:
            data = self.sampler(data)
        return data

    def __len__(self):
        return len(self.file_names)
