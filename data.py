import os.path as osp
import urllib
import gzip
import pandas as pd

from torch.utils.data import Dataset
from torchvision.transforms import transforms

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np


class IrisDataset(Dataset):
    def __init__(self, split):
        super().__init__()
        iris = load_iris()

        x = iris['data']
        y = iris['target']
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        if split == "train":
            self.x, _, self.y, _ = train_test_split(
                x, y, test_size=0.2, random_state=2
            )
        elif split == "val":
            _, self.x, _, self.y = train_test_split(
                x, y, test_size=0.2, random_state=2
            )

        self.x = self.x.astype(np.float32)

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.y)


class CoverTypeDataset(Dataset):
    MEAN = [2959.3653, 155.6568, 14.1037, 269.4282, 46.4189, 2350.1466, 212.146, 223.3187, 142.5283, 1980.2912, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    STD = [279.9845, 111.9136, 7.4882, 212.5492, 58.2952, 1559.2535, 26.7699, 19.7687, 38.2745, 1324.1941, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    def _download(self, root):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
        urllib.request.urlretrieve(url, osp.join(root, "covtype.data.gz"))
        with gzip.open(osp.join(root, "covtype.data.gz"), 'rb') as f:
            file_content = f.read()
            with open(osp.join(root, "covtype.data"), "wb") as outf:
                outf.write(file_content)

    def __init__(self, root, normalize, split):
        super().__init__()
        if not osp.isfile(osp.join(root, "covtype.data")):
            self._download()
        df = pd.read_csv(osp.join(root, "covtype.data"), header=None)

        train_indices = range(11340)
        val_indices = range(11340, 11340 + 3780)
        test_indices = range(11340 + 3780, 11340 + 3780 + 565892)

        if split == "train":
            self.df = df.iloc[train_indices]
        elif split == "val":
            self.df = df.iloc[val_indices]
        elif split == "test":
            self.df = df.iloc[test_indices]

        self.normalize = normalize
        if self.normalize:
            self.transform = lambda tensor: ((tensor - self.MEAN) / self.STD)\
                .astype(np.float32)

    def __getitem__(self, index):
        x = np.array(self.df.iloc[index][:54])
        y = self.df.iloc[index][54]
        if self.normalize:
            x = self.transform(x)
            y = y - 1
        return (x, y)

    def __len__(self):
        return len(self.df)
