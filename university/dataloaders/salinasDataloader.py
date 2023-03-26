import time

import numpy as np
import pandas as pd
import torch
from torch import device, nn
from torch.nn import init
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import random

torch.set_default_tensor_type('torch.DoubleTensor')
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)


def import_data(filename):
    """import data according to the first line: nbands nrows ncols"""
    if filename[-3:] == 'mat':
        bands_mat = loadmat(filename)
        bands_matrix = bands_mat.popitem()[1]

        nbands = bands_matrix.shape[0]
        nrows = bands_matrix.shape[1]
        # pdb.set_trace()
        if len(bands_matrix.shape) == 3:
            ncols = bands_matrix.shape[2]
        elif len(bands_matrix.shape) == 2:
            ncols = 1
        else:
            print('Incorrectbands_matrix.shape')

        # check scaling to [-1;1]
        eps = 0.1
        if abs(np.max(bands_matrix) - np.min(bands_matrix)) > eps:
            bands_matrix_new = -1 + 2 * (bands_matrix - np.min(bands_matrix)) / (np.max(bands_matrix) -
                                                                                 np.min(bands_matrix))
        else:
            bands_matrix_new = bands_matrix

        # construct 2D array
        data = np.empty([nrows * ncols, nbands])
        if len(bands_matrix.shape) > 2:
            for i in range(nrows):
                for j in range(ncols):
                    data[i * ncols + j, :] = bands_matrix_new[:, i, j]
        else:
            for i in range(nrows):
                data[i, :] = bands_matrix_new[:, i]
    else:
        f = open(filename, 'r')
        words = []
        for line in f.readlines():
            for word in line.split():
                words.append(word)
        nbands = np.int_(words[0])
        nrows = np.int_(words[1])
        ncols = np.int_(words[2])

        # training set: number of pixels with nbands -> number of inputs
        offset = 3  # offset is due to header (nbands,nrows,ncols) in words
        data = np.empty([nrows * ncols, nbands])
        if ncols > 1:
            for row in range(nrows):
                for col in range(ncols):
                    for i in range(nbands):
                        gidx = (col * nrows + row) * nbands + i + offset
                        data[row * ncols + col, i] = np.float32(words[gidx])
        else:
            for row in range(nrows):
                for i in range(nbands):
                    gidx = i * nrows + row + offset
                    data[row, i] = np.float32(words[gidx])

        f.close()
        # pdb.set_trace()
    return nbands, nrows, ncols, data


def import_labels(filename):
    """import data according to the first line: nrows ncols"""
    if filename[-3:] == 'mat':
        labels_mat = loadmat(filename)
        labels_matrix = labels_mat.popitem()[1]
        nrows = labels_matrix.shape[0]
        ncols = labels_matrix.shape[1]
        labels = labels_matrix.reshape(nrows * ncols)
    else:
        f = open(filename, 'r')
        words = []
        for line in f.readlines():
            for word in line.split():
                words.append(word)
        nrows = np.int_(words[0])
        ncols = np.int_(words[1])
        labels = np.zeros(nrows * ncols)
        offset = 2  # offset is due to header (nrows,ncols) in words
        for row in range(nrows):
            for col in range(ncols):
                labels[row + col * nrows] = np.float32(words[row + col * nrows + offset])

        f.close()
    return nrows, ncols, labels


def load_train(bandsfilename, labelsfilename):
    [nbands, nrows, ncols, data] = import_data(bandsfilename)
    [nrows, ncols, labels] = import_labels(labelsfilename)
    alldata = pd.DataFrame(data)
    alldata['labels'] = labels
    alldata['labels'] = alldata['labels'].astype('int64')

    # get data according to '0' classes
    zerodata = alldata[alldata.labels == 0]
    alldata = alldata[alldata.labels != 0]
    X_train, X_test, y_train, y_test = train_test_split(alldata.iloc[:, :-1],
                                                        alldata.iloc[:, -1],
                                                        test_size=0.2,
                                                        random_state=42)
    return nbands, nrows, ncols, X_train, X_test, y_train, y_test, zerodata


class Dataset_salinas(Dataset):
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y

    def __getitem__(self, index: int):
        # feature = self.feature[index]
        # label = self.label[index] - 1
        return self.y[index], self.X[index]
        # return torch.tensor(label, dtype=torch.long), torch.tensor(feature, dtype=torch.float32)
        return torch.tensor(self.y[index], dtype=torch.long), torch.tensor(self.X[index], dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]


def getDataLoader(*, batch_size):

    nbands, nrows, ncols, X_train, X_test, y_train, y_test, zerodata = load_train(
        '/home/wfnian/heart/高光谱_SAR数据_/salinas/salinas.txt',
        '/home/wfnian/heart/高光谱_SAR数据_/salinas/salinas_labels.txt')

    train_dataset = Dataset_salinas(np.array(X_train), np.array(y_train))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=12)
    val_dataset = Dataset_salinas(np.array(X_test), np.array(y_test))
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=12)
    return train_dataloader, val_dataloader


if __name__ == '__main__':
    getDataLoader(batch_size=10)
    pass
