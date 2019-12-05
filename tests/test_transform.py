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
# Date Created: 2019-11-26

import pytest
import numpy as np
import torch
from wavelets_pytorch.transform import WaveletTransformTorch, WaveletTransform

def test_torch_power_implementation():
    for n_channels in range(1,5):
        # create random data
        n_samples = 100
        signal_length = 42
        X = torch.rand(n_samples, n_channels, signal_length)

        # calculate power via the original numpy route and via the direct torch implementation
        wt = WaveletTransformTorch(cuda=False, channels=n_channels)
        power_np = super(WaveletTransformTorch, wt).power(X.numpy())
        power_torch = wt.power(X)

        assert np.allclose(power_np, power_torch.numpy())

def test_cwt_scipy_vs_torch_single_channel():
    # create random data
    n_samples = 100
    signal_length = 42
    X = np.random.rand(n_samples, signal_length)

    # SciPy and PyTorch based wavelet transformation
    wa_scipy = WaveletTransform(dt=1.0, dj=0.125)
    wa_torch = WaveletTransformTorch(dt=1.0, dj=0.125, cuda=False)
    cwt_scipy = wa_scipy.cwt(X)
    cwt_torch = wa_torch.cwt(X)

    # ensure that the exact same scales were used
    assert np.array_equal(wa_scipy.scales, wa_torch.scales)
    assert np.allclose(cwt_torch, cwt_scipy, rtol=1e-5, atol=1e-6)

    # test correct sizes
    assert cwt_torch.shape == (n_samples, len(wa_scipy.scales), signal_length)
    assert cwt_scipy.shape == (n_samples, len(wa_torch.scales), signal_length)

@pytest.mark.xfail(strict=True, raises=AssertionError)
def test_cwt_scipy_multi_channel():
    # create random data
    n_samples = 100
    n_channels = 12
    signal_length = 42
    X = np.random.rand(n_samples, n_channels, signal_length)

    # execute wavelet transformation
    wa = WaveletTransform(dt=1.0, dj=0.125)
    wt = wa.cwt(X)

def test_cwt_torch_multi_channel():
    # create random data
    n_samples = 100
    n_channels = 12
    signal_length = 42
    X = np.random.rand(n_samples, n_channels, signal_length)

    # execute wavelet transformation
    wa = WaveletTransformTorch(dt=1.0, dj=0.125, cuda=False, channels=n_channels)
    cwt = wa.cwt(X)
    assert cwt.shape == (n_samples, n_channels, len(wa.scales), signal_length)

class Scalogram(torch.nn.Module):
    def __init__(self, seg_width, dt, dj):
        super(Scalogram, self).__init__()
        self.wavelet = WaveletTransformTorch(dt, dj, cuda=False)
        self.wavelet.signal_length = seg_width # implicitely set num_scales

    @property
    def num_scales(self):
        return len(self.wavelet.scales)

    def forward(self, X):
        # determine scalogram for each channel
        scalograms = []
        for channel in torch.split(X, 1, dim=1):
            scalograms.append(self.wavelet.power(channel))
        X = torch.stack(scalograms, dim=1)
        return X

def test_scalogram():
    # create random data
    n_samples = 1
    n_channels = 2
    signal_length = 3
    X = torch.rand(n_samples, n_channels, signal_length)

    # determine scalogram for each channel
    scalogram_sequential = Scalogram(signal_length, dt=1.0, dj=0.125)
    scalogram_parallel = WaveletTransformTorch(dt=1.0, dj=0.125, cuda=False, channels=n_channels)
    power_sequential = scalogram_sequential(X)
    power_parallel = scalogram_parallel.power(X)

    assert np.array_equal(scalogram_sequential.wavelet.scales, scalogram_parallel.scales)
    assert torch.all(torch.eq(power_sequential, power_parallel))
