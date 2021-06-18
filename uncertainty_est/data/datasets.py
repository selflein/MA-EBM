from typing import List

from torch.utils.data import Dataset, IterableDataset


REPLACE_DATASET_DICT = {
    "celeb-a": "CelebA",
    "cifar10": "CIFAR-10",
    "fashionmnist": "FashionMNIST",
    "cifar100": "CIFAR-100",
    "gaussian_noise": "Noise",
    "lsun": "LSUN",
    "svhn": "SVHN",
    "svhn_unscaled": "OODomain",
    "uniform_noise": "Noise",
    "textures": "Textures",
    "kmnist": "KMNIST",
    "mnist": "MNIST",
    "notmnist": "NotMNIST",
    "segment-ood": "Segment OOD",
    "sensorless-ood": "Sensorless OOD",
    "segment": "Segment",
    "sensorless": "Sensorless",
    "kmnist_unscaled": "OODomain",
    "svhn_unscaled": "OODomain",
    "constant": "Constant",
}


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        super().__init__()
        self.datasets: List[Dataset] = datasets

    def __len__(self):
        return min([len(ds) for ds in self.datasets])

    def __getitem__(self, idx):
        return [ds[idx] for ds in self.datasets]


class ConcatIterableDataset(IterableDataset):
    def __init__(self, *datasets):
        super().__init__()
        self.datasets: List[Dataset] = datasets

    def __iter__(self):
        return zip(*self.datasets)
