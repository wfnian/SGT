import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class Dataset_magnet(Dataset):
    def __init__(self, flag='train', scale=False) -> None:
        assert flag in ['train', 'val'], 'not implement!'
        self.flag = flag
        ann = pd.read_csv("/home/wfnian/signal/workspace/data/C2SNR_3_6/{}_ann.txt".format(self.flag),
                          sep=',',
                          header=None)

        self.data = ann.values

    def __getitem__(self, index: int):
        val = self.data[index]
        label = int(val[0])
        seq = val[2:]

        return torch.tensor(label, dtype=torch.long), torch.tensor(seq, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)


def getDataLoader(*, batch_size):

    train_dataset = Dataset_magnet(flag='train')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = Dataset_magnet(flag='val')
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader


if __name__ == '__main__':
    train_dataset = Dataset_magnet(flag='train')
    print(train_dataset[0])
    # print(len(set(train_dataset[0][0].numpy())))
