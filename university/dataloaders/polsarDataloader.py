import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class Dataset_india(Dataset):
    def __init__(self, flag='train') -> None:
        assert flag in ['train', 'val'], 'not implement!'
        self.flag = flag
        feature = pd.read_csv(
            "/home/wfnian/heart/高光谱_SAR数据_/PolSAR-Flevoland_随机取样1%_109通道/{}_feature.txt".format(self.flag), header=None, sep="	")
        label = pd.read_csv(
            "/home/wfnian/heart/高光谱_SAR数据_/PolSAR-Flevoland_随机取样1%_109通道/{}_label.txt".format(self.flag), header=None)
        self.feature = feature.values
        self.label = label.values[:, 0]
        print(self.label.shape)
        print(self.feature.shape)

    def __getitem__(self, index: int):
        feature = self.feature[index]
        label = self.label[index]-1

        return torch.tensor(label, dtype=torch.long), torch.tensor(feature, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.feature)


def getDataLoader(*, batch_size):

    train_dataset = Dataset_india(flag='train')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    val_dataset = Dataset_india(flag='val')
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=12)

    return train_dataloader, val_dataloader


if __name__ == '__main__':
    train_dataset = Dataset_india(flag='train')
    print(train_dataset[0])
    # print(len(set(train_dataset[0][0].numpy())))
