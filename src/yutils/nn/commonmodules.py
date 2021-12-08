import torch
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy

from .nnconfig import NNConfig

class BaseModule(LightningModule):
    def __init__(self, data_provider, model, config):
        super().__init__()
        self.model = model
        self.data_provider = data_provider
        self.config = config

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = NNConfig.get_optimizer(self.config, self.parameters())
        scheduler = None
        if 'scheduler' in self.config:
            scheduler = NNConfig.get_scheduler(
                self.config, 
                optimizer)
        result = {}
        result["optimizer"]=optimizer

        if scheduler!= None:
            if 'ReduceLROnPlateau' in self.config.scheduler:
                result["lr_scheduler"]={ 
                    "scheduler": scheduler, 
                    "monitor": self.config.scheduler.extra.monitor
                }
        return result

    def prepare_data(self):
        self.data_provider.prepare_data()

    def train_dataloader(self):
        return self.data_provider.train_dataloader()

    def val_dataloader(self):
        return self.data_provider.val_dataloader()

    def test_dataloader(self):
        return self.data_provider.test_dataloader()

class ClassificationModule(BaseModule):
    def __init__(self, data_provider, model, config):
        super().__init__(data_provider, model, config)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log("val_acc", acc, prog_bar=True)

        return loss
    
    def validation_epoch_end(self, outputs):
        loss = torch.stack(outputs).mean()
        self.log("val_loss", loss)        
