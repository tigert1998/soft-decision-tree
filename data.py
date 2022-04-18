from torch.utils.data import Dataset

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.y)
