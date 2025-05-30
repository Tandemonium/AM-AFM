
import numpy as np
import torch

from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.environments import LightningEnvironment

from sklearn import model_selection as ms
from torch.utils.data import DataLoader, Dataset


class AFMDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: list|None = None, expand: int|None = None):
        if expand is not None:
            inputs = np.expand_dims(inputs, axis=expand)
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        if targets is not None:
            targets = np.expand_dims(targets, axis=-1)
            self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self, 'targets'):
            return self.inputs[idx], self.targets[idx]
        else:
            return self.inputs[idx]


class Wrapper(LightningModule):
    def __init__(self, model: type[torch.nn.Module], criterion: torch.nn, optimizer: type[torch.optim.Optimizer], 
                 in_channels: int, out_channels: int, learning_rate: float = 1e-3, 
                 weight_decay: float = 0.0, **kwargs):
        super().__init__()
        self.criterion = criterion
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = model(in_channels, out_channels, **kwargs)
    
    def __call__(self, *args, **kwds) -> torch.Tensor:
        return super().__call__(*args, **kwds)
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)
    
    def step(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, targets = batch
        pred = self.forward(inputs)
        loss = self.criterion(pred, targets)
        return pred, targets, loss
    
    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        pred, targets, loss = self.step(batch)
        return loss
    
    def validation_step(self, batch: torch.Tensor) -> torch.Tensor:
        pred, targets, loss = self.step(batch)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        optimizer = self.optimizer(params=self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return {"optimizer": optimizer}


class NeuralNetworkClassifier:
    def __init__(self, name: str, model: type[torch.nn.Module], save_dir: str, in_channels: int, 
                 out_channels: int, criterion: torch.nn = torch.nn.BCEWithLogitsLoss(), 
                 optimizer: type[torch.optim.Optimizer] = torch.optim.AdamW, learning_rate: float = 1e-4, 
                 weight_decay: float = 1e-05, batch_size: int = 256, val_size: float = 0.2, 
                 expand: int|None = None, num_workers: int = 4, max_epochs: int = 100, device_type: str = 'gpu', 
                 num_devices: int = 1, seed: int = 42, **kwargs):
        self.name = name
        self.batch_size = batch_size
        self.val_size = val_size
        self.expand = expand
        self.num_workers = num_workers
        self.max_epochs = max_epochs
        self.device_type = device_type
        self.num_devices = num_devices
        self.seed = seed
        self.is_fitted: bool = False
        self.n_features_in_: int = None

        logger = TensorBoardLogger(save_dir, name=None)
        callbacks = [ModelCheckpoint(save_top_k=1, monitor='val_loss', mode='min', verbose=False, save_last=False), 
                     EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=False)]
        self.trainer = Trainer(strategy='auto', plugins=[LightningEnvironment()], max_epochs=self.max_epochs, 
                               accelerator=self.device_type, devices=self.num_devices, callbacks=callbacks, logger=logger,
                               log_every_n_steps=1)
        self.wrapper = Wrapper(model, criterion, optimizer, in_channels, out_channels, learning_rate, weight_decay, 
                               **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.n_features_in_ = X.shape[1]
        X_train, X_val, y_train, y_val = ms.train_test_split(X, y, test_size=self.val_size, random_state=self.seed)
        train_loader = DataLoader(AFMDataset(X_train, y_train, self.expand), batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, persistent_workers=bool(self.num_workers), 
                                  pin_memory=True)
        val_loader = DataLoader(AFMDataset(X_val, y_val, self.expand), batch_size=self.batch_size, 
                                num_workers=self.num_workers, persistent_workers=bool(self.num_workers), 
                                pin_memory=True)
        self.trainer.fit(self.wrapper, train_loader, val_loader)
        self.is_fitted = True

    def predict(self, X: np.ndarray):
        dataset = AFMDataset(X, expand=self.expand)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
                                persistent_workers=bool(self.num_workers), pin_memory=True)
        pred = self.trainer.predict(self.wrapper, dataloader)
        pred = torch.concat(pred, dim=0)
        pred = torch.round(torch.sigmoid(pred)).squeeze()
        pred = pred.cpu().numpy()
        return pred
