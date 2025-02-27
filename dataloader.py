import torch
import numpy as np
from torch.utils.data import Dataset


def replace_with_avg(arr, ratio):
    mask = np.random.rand(*arr.shape[:2]) < ratio
    masked_arr = np.copy(arr)
    masked_arr[mask] = np.nan

    for i in range(2, arr.shape[1] - 2):
        masked_arr[:, i] = np.nanmean(arr[:, i - 2 : i + 3], axis=1)

    return masked_arr


class MyDataset(Dataset):
    def __init__(self, data_path, labels_path):

        self.data = np.load(data_path)
        self.labels = np.load(labels_path)
        self.data = self.data[:, :, :]
        self.labels = self.labels[:, :, :3]
        self.data = self.data.astype(float)
        self.labels = self.labels.astype(float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data, label
