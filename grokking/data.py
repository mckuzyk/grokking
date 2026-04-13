from torch.utils.data import Dataset
import torch


torch.manual_seed(42)


class ModPDataset(Dataset):
    def __init__(self, P=113):
        self.P = P
        a, b = torch.meshgrid(torch.arange(P), torch.arange(P), indexing="ij")
        a = torch.flatten(a)
        b = torch.flatten(b)
        c = (a + b) % P
        eq_token = torch.ones_like(a) * (P)

        self.x = torch.stack((a, b, eq_token), dim=1)
        self.y = c

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.P**2


def train_test_split(dset, train_frac=0.3):
    idx = torch.randperm(len(dset))
    stop = int(train_frac * len(dset))
    train_inputs, train_labels = dset[idx[:stop]]
    test_inputs, test_labels = dset[idx[stop:]]
    return train_inputs, train_labels, test_inputs, test_labels
