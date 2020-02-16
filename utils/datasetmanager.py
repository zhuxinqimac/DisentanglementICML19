import os
import sys
import h5py
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from config.path import DSPRITESPATH, SHAPES3DPATH
from utils.reader_op import read_npy, read_npy_py3
from utils.datamanager import DspritesManager, Shapes3DManager

import numpy as np

def dsprites_manager():
    # dataset_zip = read_npy(DSPRITESPATH+'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', encoding='latin1')
    dataset_zip = read_npy_py3(DSPRITESPATH+'dsprites_py3.npz')
    dm = DspritesManager(dataset_zip)
    return dm

def shapes_3d_manager():
    dataset_zip = h5py.File(SHAPES3DPATH+'3dshapes.h5', 'r')
    dm = Shapes3DManager(dataset_zip)
    return dm

