from .dataprovider import DataProvider

from torchvision import datasets

from torch.utils.data import random_split

# Some basic data providers
class MNISTDataProvider(DataProvider):
    def __init__(self, root_dir, image_transform, batch_size):
        super().__init__(root_dir, image_transform, batch_size)

    def prepare_data(self):
        self.test_dataset = datasets.MNIST(
            root=self.root_dir,
            train=False,
            download=True,
            transform=self.image_transform,
        )
        train_dataset = datasets.MNIST(
            root=self.root_dir,
            train=True,
            download=True,
            transform=self.image_transform,
        )
        self.train_dataset, self.val_dataset = random_split(
            train_dataset, [55000, 5000]
        )


class FashionMNISTDataProvider(DataProvider):
    def __init__(self, root_dir, transform, batch_size):
        super().__init__(root_dir, transform, batch_size)

    def prepare_data(self):
        self.test_dataset = datasets.FashionMNIST(
            root=self.root_dir,
            train=False,
            download=True,
            transform=self.image_transform,
        )
        train_dataset = datasets.FashionMNIST(
            root=self.root_dir,
            train=True,
            download=True,
            transform=self.image_transform,
        )
        self.train_dataset, self.val_dataset = random_split(
            train_dataset, [55000, 5000]
        )


class Cifar10DataProvider(DataProvider):
    def __init__(self, root_dir, image_transform):
        super().__init__(root_dir, image_transform)

    def prepare_data(self):
        self.test_dataset = datasets.CIFAR10(
            root=self.root_dir,
            train=False,
            download=True,
            transform=self.image_transform,
        )
        train_dataset = datasets.CIFAR10(
            root=self.root_dir,
            train=True,
            download=True,
            transform=self.image_transform,
        )
        train_dataset_length = len(train_dataset)
        split = [int(0.9 * train_dataset_length), int(0.1 * train_dataset_length)]
        self.train_dataset, self.val_dataset = random_split(train_dataset, split)
