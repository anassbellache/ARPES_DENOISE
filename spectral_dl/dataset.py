from genericpath import isfile
from pathlib import Path
import numpy as np
import torch
import astra
import os
from arpys import dl, pp
from torch.utils.data import Dataset

class ARPESDataset(Dataset):
    def __init__(self, paths, flux, transforms=None):
        super().__init__()
        files = [f for f in os.listdir(paths) if os.path.isfile(os.path.join(paths, f))]
        files.sort()
        self.files = files[::2]
        self.data = []
        for f in self.files:
            D = dl.load_data(os.path.join(paths, f))
            self.data.append(D.data[0])
        self.data = np.array(self.data)
        self.flux = flux
        self.transforms = transforms
    
    def __getitem__(self, i):
        clean_graph = self.data[i,:,:]
        noisy_graph = astra.add_noise_to_sino(clean_graph, self.flux)
        if self.transforms is not None:
            noisy_graph = self.transforms(noisy_graph)
            clean_graph = self.transforms(self.data[i,:,:])
        return noisy_graph, clean_graph
    
    def __len__(self):
        return self.data.shape[0]


class NPYArpes(Dataset):
    def __init__(self, np_file_path, transforms=None):
        super().__init__()
        files = np_file_path
        self.data = np.load(files[0])
        self.noise = np.load(files[1])
        self.transforms = transforms
    
    def __getitem__(self, index):
        clean_graph = self.data[index,:,:]
        noisy_graph = self.noise[index,:,:]
        if self.transforms is not None:
            clean_graph = self.transforms(self.data[index,:,:])
            noisy_graph = self.transforms(self.noise[index,:,:])
        return noisy_graph, clean_graph
    
    def __len__(self):
        return len(self.data)
