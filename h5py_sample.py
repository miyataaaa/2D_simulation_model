# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 11:49:11 2021

@author: miyar
"""

import numpy as np
import h5py

with h5py.File('file_open_sample.hdf5', mode='w') as f:
    sample = np.ones((2, 3), dtype=float)
    pass