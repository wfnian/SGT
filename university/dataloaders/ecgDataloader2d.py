import json
import time
from random import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from albumentations import Compose, Normalize
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchsampler import ImbalancedDatasetSampler as IDS

augment = Compose([Normalize(), ToTensorV2()])


class EcgDataset1D(Dataset):
    def __init__(self, ann_path):
        super().__init__()
        self.flag = ann_path.split('/')[-1].split('.')[0]
        self.data = json.load(open(ann_path))
        self.mapper = json.load(open("/home/wfnian/heart/ecg-classification/data/class-mapper.json"))

    def get_labels(self):
        labels = []
        for i in self.data:
            labels.append(self.mapper[i['label']])
        return labels

    def __getitem__(self, index):
        img = cv2.imread(self.data[index]["path"])
        img = augment(**{"image": img})["image"]

        return self.mapper[self.data[index]["label"]], img
        return {"image": img, "class": self.mapper[self.data[index]["label"]]}

    def __len__(self):
        return len(self.data)


def getDataLoader(*, batch_size):

    train_dataset = EcgDataset1D('/home/wfnian/heart/ecg-classification/data/train.json')

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = EcgDataset1D('/home/wfnian/heart/ecg-classification/data/val.json')
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader


if __name__ == '__main__':
    train_dataset = EcgDataset1D('/home/wfnian/heart/ecg-classification/data/train.json')
    # train_dataloader = DataLoader(train_dataset, sampler=IDS(train_dataset), batch_size=1024)
    print(train_dataset[0][1].shape)
    print()
    # print(len(set(train_dataset[0][0].numpy())))
