# MIT License
# 
# Copyright (c) 2019 Matthias Gazzari
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Matthias Gazzari
# Date Created: 2019-11-20

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from wavelets_pytorch.transform import WaveletTransformTorch

class Scalogram(torch.nn.Module):
    def __init__(self, seg_width, dt=0.05, dj=0.125, cuda=True):
        super(Scalogram, self).__init__()
        self.cuda = cuda
        self.wavelet = WaveletTransformTorch(dt, dj, cuda=cuda)
        self.wavelet.signal_length = seg_width # implicitely set num_scales

    @property
    def num_scales(self):
        return len(self.wavelet.scales)

    def forward(self, X):
        # move to cpu if using cuda
        if self.cuda:
            X = X.cpu()
        # determine scalogram for each channel
        scalograms = []
        for channel in torch.split(X, 1, dim=1):
            power_np = self.wavelet.power(channel.numpy())
            power_torch = torch.tensor(power_np).float()
            scalograms.append(power_torch)
        X = torch.stack(scalograms, dim=1)
        # move back to gpu if using cuda
        if self.cuda:
            X = X.cuda()
        return X

class ConvNet(torch.nn.Module):
    def __init__(self, channels, segment_width, conv_filters=64, kernel_size=3):
        super(ConvNet, self).__init__()
        scalogram = Scalogram(segment_width, cuda=torch.cuda.is_available())
        self.sequence = torch.nn.Sequential(
            scalogram,
            torch.nn.ConstantPad2d((kernel_size - 1, 0, kernel_size - 1, 0), 0),
            torch.nn.Conv2d(
                in_channels=channels,
                out_channels=conv_filters,
                kernel_size=(kernel_size, kernel_size),
            ),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(
                in_features=conv_filters * scalogram.num_scales * segment_width,
                out_features=1
            ),
        )

    def forward(self, X):
        return self.sequence(X).reshape(-1)

def main(n_samples=1000, n_channels=42, segment_width=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create random data
    X = torch.rand(n_samples, n_channels, segment_width, device=device)
    y = torch.rand(n_samples, device=device)
    data_loader = DataLoader(TensorDataset(X, y), batch_size=4)

    net = ConvNet(n_channels, segment_width).to(device)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters())

    for epoch in range(2):
        print("Epoch %d" % epoch)
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    main()
