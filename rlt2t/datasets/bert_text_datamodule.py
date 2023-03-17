import os
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from rlt2t.datasets.bert_text_dataset import BertTextDataset

class BertTextDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 16,
                 max_token_len: int = 128,
                 start_idx: int = 106,
                 num_words: int = 7000,
                 eos_id: int = 105,
                 num_workers: int = 0,
                 pin_memory: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.start_idx = start_idx
        self.num_words = num_words
        self.eos_id = eos_id
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning separately when using `trainer.fit()` and `trainer.test()`!
        The `stage` can be used to differentiate whether the `setup()` is called before trainer.fit()` or `trainer.test()`."""
        if not self.data_train or not self.data_val or not self.data_test:
            self.data_train = BertTextDataset(os.path.join(self.data_dir, 'train.json'),
                                              max_token_len=self.max_token_len,
                                              start_idx=self.start_idx,
                                              num_words=self.num_words,
                                              eos_id=self.eos_id)
            self.data_val = BertTextDataset(os.path.join(self.data_dir, 'val.json'),
                                            max_token_len=self.max_token_len,
                                            start_idx=self.start_idx,
                                            num_words=self.num_words,
                                            eos_id=self.eos_id)
            self.data_test = BertTextDataset(os.path.join(self.data_dir, 'test.json'),
                                             max_token_len=self.max_token_len,
                                             start_idx=self.start_idx,
                                             num_words=self.num_words,
                                             eos_id=self.eos_id)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
