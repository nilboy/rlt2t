from typing import Any, List
import torch
from torch import nn
from pytorch_lightning import LightningModule
from torch.optim import SGD

class Rank2Model(LightningModule):
    def __init__(self,
                 lr: float = 1e-5,
                 hidden_num: int = 4):
        super(Rank2Model, self).__init__()
        self.save_hyperparameters()

        self.weight = nn.Parameter(torch.FloatTensor([1.0] * hidden_num))

    def forward(self, x1=None, x2=None):
        y1 = torch.sum(torch.abs(self.weight) * x1, axis=1)
        y2 = torch.sum(torch.abs(self.weight) * x2, axis=1)
        diff_y = y1 - y2
        loss = -torch.log(torch.sigmoid(diff_y)).sum()
        acc = torch.sum(diff_y > 0)/diff_y.shape[0]
        return loss, acc

    def step(self, batch: Any):
        loss, acc = self.forward(batch['x1'],
                              batch['x2'])
        return loss, acc

    def training_step(self,
                      batch: Any,
                      batch_idx: int):
        loss, acc = self.step(batch)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train/acc', acc, on_step=True, on_epoch=True, prog_bar=False)
        return {
            'loss': loss
        }

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        print(self.weight)
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, acc = self.step(batch)
        self.log('val/loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('val/acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return {
            'loss': loss
        }

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self) -> Any:
        optimizer = SGD(self.parameters(), lr=self.hparams.lr)
        return optimizer