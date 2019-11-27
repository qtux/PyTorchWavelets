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

import numpy as np
import torch
from wavelets_pytorch.transform import WaveletTransformTorch, WaveletTransform

def test_torch_power_implementation():
    # create random data
    n_samples = 100
    n_channels = 1
    signal_length = 42
    X = torch.rand(n_samples, n_channels, signal_length)

    # calculate power via the original numpy route and via the direct torch implementation
    wt = WaveletTransformTorch(cuda=False)
    power_np = super(WaveletTransformTorch, wt).power(X.numpy())
    power_torch = wt.power(X)

    assert np.allclose(power_np, power_torch.numpy())

def test_cwt_scipy_vs_torch():
    # create random data
    n_samples = 100
    signal_length = 42
    X = np.random.rand(n_samples, signal_length)

    # calculate power via the original numpy route and via the torch implementation
    wa_scipy = WaveletTransform(dt=1.0, dj=0.125)
    wa_torch = WaveletTransformTorch(dt=1.0, dj=0.125, cuda=False)

    cwt_scipy = wa_scipy.cwt(X)
    cwt_torch = wa_torch.cwt(X)

    # ensure that the exact same scales were used
    assert np.array_equal(wa_scipy.scales, wa_torch.scales)
    assert np.allclose(cwt_torch, cwt_scipy, rtol=1e-5, atol=1e-6)
