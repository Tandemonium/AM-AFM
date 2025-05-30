import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn_crfsuite import CRF as BaseCRF
from torch.autograd import Variable


class CRF(BaseCRF):
    def x_transform(self, X: np.ndarray) -> list[list[dict[str, str]]]:
        return [list(pd.DataFrame(X, columns=np.arange(X.shape[1]).astype(str)).to_dict('index').values())]
    
    def y_transform(self, y: np.ndarray) -> list[list[str]]:
        return [y.astype(str).tolist()]

    def fit(self, X: np.ndarray, y: np.ndarray, X_dev=None, y_dev=None):
        X = self.x_transform(X)
        y = self.y_transform(y)
        return super().fit(X, y, X_dev, y_dev)
    
    def predict(self, X) -> np.ndarray:
        X = self.x_transform(X)
        pred = super().predict(X)
        return pred.flatten().astype(bool)


class CNN(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, hidden_dims: list[int], input_len: int,
                 activation: nn.Module = nn.ReLU, norm: bool = False, dropout: float|None = None, 
                 conv_kernel_size: int = 3, conv_stride: int = 1, 
                 conv_padding: int = 1, pool_kernel_size: int = 2, pool_stride: int|None = None, pool_padding: int = 0):
        super().__init__()
        channels = [input_channels] + hidden_dims
        layers = []
        for i in range(len(channels) - 2):
            layers.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=conv_kernel_size, 
                                    stride=conv_stride, padding=conv_padding))
            if norm:
                layers.append(nn.BatchNorm1d(channels[i + 1]))
            layers.append(activation())
            layers.append(nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding))
        layers.append(nn.Flatten())
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dims[-2] * (input_len // pool_kernel_size**(len(hidden_dims) - 1)), hidden_dims[-1]))
        if norm:
            layers.append(nn.BatchNorm1d(hidden_dims[-1]))
        layers.append(activation())
        layers.append(nn.Linear(hidden_dims[-1], num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class LSTM(nn.Module):
    def __init__(self, input_size: int, num_classes: int, lstm_size: int, n_lstm_layers: int, fc_size: int, 
                 activation: nn.Module = nn.ReLU, dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_size, num_layers=n_lstm_layers)
        self.fc1 = nn.Linear(lstm_size, fc_size)
        self.fc2 = nn.Linear(fc_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation()
        self.n_lstm_layers = n_lstm_layers
        self.lstm_size = lstm_size

    def forward(self, x: torch.Tensor):
        h_0 = Variable(torch.zeros(self.n_lstm_layers, self.lstm_size).to(x.device))
        c_0 = Variable(torch.zeros(self.n_lstm_layers, self.lstm_size).to(x.device))
        x, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        x = self.activation(x)
        # x = self.dropout(x)
        x = self.fc1(x)
        # x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ResNetBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super(ResNetBlock, self).__init__()
        self.conv_1 = torch.nn.Conv2d(
            in_channels, in_channels, 3, stride=1, 
            padding='same', padding_mode='zeros')
        self.conv_2 = torch.nn.Conv2d(
            in_channels, in_channels, 3, stride=1, 
            padding='same', padding_mode='zeros')
        self.bn = torch.nn.BatchNorm2d(in_channels)
    
    def forward(self, x):
        input_val = x
        x = self.bn(x)
        x = self.conv_1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv_2(x)
        return input_val + x


class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv_1 = torch.nn.Conv2d(
            3, 16, 5, stride=1, 
            padding='same', padding_mode='zeros')
        self.bn_1 = torch.nn.BatchNorm2d(16)
        self.block_1 = ResNetBlock(16)
        self.pool_1 = torch.nn.MaxPool2d(2)

        self.conv_2 = torch.nn.Conv2d(
            16, 32, 5, stride=1, 
            padding='same', padding_mode='zeros')
        self.bn_2 = torch.nn.BatchNorm2d(32)
        self.block_2 = ResNetBlock(32)
        self.pool_2 = torch.nn.MaxPool2d(2)

        self.drop_1 = torch.nn.Dropout(0.5)
        self.linear_1 = torch.nn.Linear(2048, 512)
        self.bn_3 = torch.nn.BatchNorm1d(512)
        self.linear_2 = torch.nn.Linear(512, 10)
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.block_1(x)
        x = self.pool_1(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.block_2(x)
        x = self.pool_2(x)

        x = torch.flatten(x, start_dim=1)
        x = self.drop_1(x)
        x = self.linear_1(x)
        x = self.bn_3(x)
        x = torch.nn.functional.relu(x)
        return self.linear_2(x)
