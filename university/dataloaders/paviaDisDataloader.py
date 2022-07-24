import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class Dataset_india(Dataset):
    def __init__(self, flag='train') -> None:
        assert flag in ['train', 'val'], 'not implement!'
        self.flag = flag
        feature = pd.read_csv("/home/wfnian/heart/高光谱_SAR数据_王方年/高光谱空间不相交数据/论文对比/Pavia/{}_feature.txt".format(
            self.flag),
                              header=None,
                              sep="	")
        label = pd.read_csv("/home/wfnian/heart/高光谱_SAR数据_王方年/高光谱空间不相交数据/论文对比/Pavia/{}_label.txt".format(self.flag),
                            header=None)
        self.feature = feature.values
        self.label = label.values[:, 0]
        print(self.label.shape)
        print(self.feature.shape)

    def __getitem__(self, index: int):
        feature = self.feature[index]
        label = self.label[index] - 1
        if self.flag == 'train':
            # if label == 0 or label == 2 or label == 5:
            # if random.randint(1, 2) == 1:
            feature = feature  #+ random.random()

        return torch.tensor(label, dtype=torch.long), torch.tensor(feature, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.feature)


def getDataLoader(*, batch_size):

    train_dataset = Dataset_india(flag='train')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    val_dataset = Dataset_india(flag='val')
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=12)

    return train_dataloader, val_dataloader


if __name__ == '__main__':
    train_dataset = Dataset_india(flag='val')
    print(train_dataset[0][1].shape)
    c = train_dataset[1]
    import matplotlib.pyplot as plt

    print()
