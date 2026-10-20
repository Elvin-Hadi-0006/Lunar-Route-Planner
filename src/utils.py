import utils
import numpy as np
from scipy import interpolate
from scipy import integrate
from scipy import optimize
import rasterio
from rasterio.windows import Window
from pathlib import Path
import pvl
import matplotlib as mpl
from matplotlib import pyplot as plt

def read_data(lbl_file_path, window:Window):
    label = pvl.load(lbl_file_path)
    with rasterio.open(lbl_file_path) as file:
        data = file.read(1, window=window) * label['UNCOMPRESSED_FILE']['IMAGE']['SCALING_FACTOR'] * 1000 # get elevation model in meters
        bounds = file.bounds
        x_lin = np.linspace(0, bounds.right-bounds.left, data.shape[0])
        y_lin = np.linspace(0, bounds.bottom-bounds.top, data.shape[1])

    return data, x_lin, y_lin

def downsample(arr, n):
    """Downsample a 2d array by averaging non-overlapping n x n blocks."""
    h, w = arr.shape
    h_crop = h - (h % n)
    w_crop = w - (w % n)
    arr = arr[:h_crop, :w_crop]
    return arr.reshape(h_crop // n, n, w_crop // n, n).mean(axis=(1, 3))

def directional_gradient(
        pos:np.ndarray, 
        dir:np.ndarray, 
        x_grad_interp:interpolate.RegularGridInterpolator, 
        y_grad_interp:interpolate.RegularGridInterpolator
    ):
    """
    returns gradient tangent and perpendicular to direction vector
    """
    if pos.ndim==1:
        pos = np.atleast_2d(pos).T

    if dir.ndim==1:
            dir = np.atleast_2d(dir).T

    normalDir = np.stack([-dir[:, 1], dir[:, 0]], axis=-1) # (N,2)

    grad = np.stack([x_grad_interp(pos), y_grad_interp(pos)], axis=-1) # (N,2) 

    forward = np.sum(grad * dir, axis=-1)               # (N,)
    sideways = np.sum(grad * normalDir, axis=-1)              # (N,)
    return forward, sideways

def get_index(query, vec):
    i = np.arange(vec.size)
    return int(np.rint(np.interp(query, vec, i)))