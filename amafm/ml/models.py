import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, hidden_dims: list[int], input_len: int,
                 activation: nn.Module = nn.ReLU, conv_kernel_size: int = 3, conv_stride: int = 1, 
                 conv_padding: int = 1, pool_kernel_size: int = 2, pool_stride: int|None = None, pool_padding: int = 0):
        super().__init__()
        channels = [input_channels] + hidden_dims
        layers = []
        for i in range(len(channels) - 2):
            layers.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=conv_kernel_size, 
                                    stride=conv_stride, padding=conv_padding))
            layers.append(activation())
            layers.append(nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(hidden_dims[-2] * (input_len // pool_kernel_size**(len(hidden_dims) - 1)), hidden_dims[-1]))
        layers.append(activation())
        layers.append(nn.Linear(hidden_dims[-1], num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


import torch
from torch.autograd import Variable

class biLSTM(nn.Module):
    def __init__(self, input_size, lstm_size, hidden_size, lstm_layers, activation, dropout):
        super().__init__()
        self.num_directions = 2
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=lstm_size,
                            num_layers=lstm_layers,
                            bidirectional=True)
        self.fc1 = nn.Linear(lstm_size * self.num_directions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation()
        self.sigmoid = nn.Sigmoid()
        self.lstm_layers = lstm_layers
        self.lstm_size = lstm_size

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.lstm_layers * self.num_directions, self.lstm_size).to('gpu'))
        c_0 = Variable(torch.zeros(self.lstm_layers * self.num_directions, self.lstm_size).to('gpu'))
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        out = self.activation(out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))
        return out
