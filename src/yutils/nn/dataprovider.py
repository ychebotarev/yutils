from torch.utils.data import DataLoader


class DataProvider:
    def __init__(self, root_dir, image_transform, batch_size=256):
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.image_transform = image_transform

    def prepare_data(self):
        raise NotImplemented

    def get_data_loader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size)

    def train_dataloader(self):
        return self.get_data_loader(self.train_dataset)

    def val_dataloader(self):
        return self.get_data_loader(self.val_dataset)

    def test_dataloader(self):
        return self.get_data_loader(self.test_dataset)
