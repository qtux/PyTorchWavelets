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
from wavelets_pytorch.transform import WaveletTransformTorch

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
