import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torchaudio
from pathlib import Path


class Dataset_india(Dataset):
    def __init__(self, flag='train', scale=False) -> None:
        assert flag in ['train', 'val'], 'not implement!'
        self.flag = flag
        self.path = Path("/home/wfnian/audio/esc/audio/")
        assert self.path.exists()
        if self.flag == 'train':
            self.files = list(self.path.glob("[1534]*.wav"))  # +list(self.path.glob("[2]*.wav"))[:200]
        else:
            self.files = list(self.path.glob("[2]*.wav"))  # [200:]

    def __getitem__(self, index: int):

        wav, sr = torchaudio.load(self.files[index])
        wav = torchaudio.transforms.Resample(sr, 1000)(wav)

        # mfcc = torchaudio.compliance.kaldi.mfcc(wav).t()

        val = wav.squeeze()  #mfcc  # [0]
        # tensor_minusmean = val - val.mean()
        # val = tensor_minusmean/tensor_minusmean.abs().max()

        label = int(self.files[index].stem.split('-')[-1])

        return torch.tensor(label, dtype=torch.long), val

    def __len__(self) -> int:
        return len(self.files)


def getDataLoader(*, batch_size):

    train_dataset = Dataset_india(flag='train')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    val_dataset = Dataset_india(flag='val')
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=12)

    return train_dataloader, val_dataloader


if __name__ == '__main__':
    train_dataset = Dataset_india(flag='train')
    print(train_dataset[0][1].shape)
