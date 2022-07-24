import time
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchsampler import ImbalancedDatasetSampler as IDS
import matplotlib.pyplot as plt
from random import random


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
        img = np.load(self.data[index]["path"]).astype("float32")
        # #~ 数据增强
        # if self.flag == 'train':
        #     if self.mapper[self.data[index]["label"]] == 6 or self.mapper[self.data[index]["label"]] == 3:
        #         if random() > 0.618:
        #             img = img + img * (random() * 0.8 - 0.4)
        # img = img.reshape(1, img.shape[0])
        # #~数据增强

        return torch.tensor(self.mapper[self.data[index]["label"]], dtype=torch.long), torch.tensor(img)
        return {"image": img, "class": self.mapper[self.data[index]["label"]]}

    def __len__(self):
        return len(self.data)


def getDataLoader(*, batch_size):

    train_dataset = EcgDataset1D('/home/wfnian/heart/ecg-classification/data/train.json')
    # weights = [
    #     0.09 if label == 0 else 0.01 if label == 1 else 0.09 if label == 2 else 0.09 if label ==
    #     3 else 0.12 if label == 4 else 1.00 if label == 5 else 0.20 if label == 6 else 7
    #     for label, data in train_dataset]
    # sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_dataset), replacement=True)
    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=sampler)
    # train_dataloader = DataLoader(train_dataset, sampler=IDS(train_dataset), batch_size=batch_size)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = EcgDataset1D('/home/wfnian/heart/ecg-classification/data/val.json')
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader


if __name__ == '__main__':
    train_dataset = EcgDataset1D('/home/wfnian/heart/ecg-classification/data/train.json')
    train_dataloader = DataLoader(train_dataset, sampler=IDS(train_dataset), batch_size=1024)
    print(train_dataset[0])
    print()
    # print(len(set(train_dataset[0][0].numpy())))
