import time
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
import torch
import numpy as np
import random
from torchvision import transforms
from torch import nn
from tqdm import tqdm

transform = transforms.Compose([
    # transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class Dataset_magnet_pic(Dataset):
    def __init__(self, txt_path, transform=transform):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()  # 默认删除的是空白符（'\n', '\r', '\t', ' '）
            words = line.split()  # 默认以空格、换行(\n)、制表符(\t)进行分割，大多是"\"
            imgs.append((words[0], int(words[1])))  # 存放进imgs列表中

        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]  # fn代表图片的路径，label代表标签
        # fn = fn.replace('val', 'val_wvd')
        # fn = fn.replace('train', 'train_wvd')
        img = Image.open(fn).convert('RGB')
        # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
        if self.transform is not None:
            img = self.transform(img)  # 在这里做transform，转为tensor等等

        return label, img

    def __len__(self):
        return len(self.imgs)


def getDataLoader(*, batch_size):

    train_dataset = Dataset_magnet_pic(txt_path="/home/wfnian/signal/workspace/CNNS/train_3_6.txt")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = Dataset_magnet_pic(txt_path='/home/wfnian/signal/workspace/CNNS/val_3_6.txt')
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader


if __name__ == '__main__':
    train_dataset = Dataset_magnet_pic(txt_path="/home/wfnian/signal/workspace/CNNS/val_3_6.txt")
    print(train_dataset[0])
    # print(len(set(train_dataset[0][0].numpy())))
