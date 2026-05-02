import h5py
import torch
from torch.utils.data import Dataset

class CarlaDataset(Dataset):
    def __init__(self, hdf5_path):
        self.file = h5py.File(hdf5_path, 'r')
        self.obs = self.file["observations"]
        self.actions = self.file["actions"]

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        state = torch.tensor(self.obs[idx], dtype=torch.float32)
        action = torch.tensor(self.actions[idx], dtype=torch.float32)
        return state, action
