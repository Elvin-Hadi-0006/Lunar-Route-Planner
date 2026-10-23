import numpy as np
from scipy import interpolate
import rasterio
from rasterio.windows import Window
from pathlib import Path
import pvl

class SLDEM_reader():
    """
    Reads an SLDEM lunar elevation map

    Parameters
    ----------
    lbl_file_path : pathlib.Path
        Path for the .lbl file of the data
    min_longitude : float
        The minimum longitude of the viewing window
    max_longitude : float
        The maximum longitude of the viewing window
    min_latitude : float
        The minimum latitude of the viewing window
    max_latitude : float
        The minimum latitude of the viewing window

    Attributes
    ----------
    min_longitude : float
        The minimum longitude of the viewing window
    max_longitude : float
        The maximum longitude of the viewing window
    min_latitude : float
        The minimum latitude of the viewing window
    max_latitude : float
        The minimum latitude of the viewing window
    elevation : numpy.ndarray
        2D array containing the elevation map in meters.
        x axis is in the column-wise direction and the y axis is in the row-wise direction.
    x : numpy.ndarray
        1D array containing the coordinates in the west-east direction in meters. 
        0 at the westernmost point in the viewing window.
    y : numpy.ndarray
        1D array containing the coordinates in the north-south direction in meters.
        0 at the northenmost point in the viewing window.

    """

    def __init__(self, 
                 lbl_file_path:Path, 
                 min_longitude:float, 
                 max_longitude:float, 
                 min_latitude:float, 
                 max_latitude:float):
        if lbl_file_path.suffix != '.lbl':
            raise Exception('File must have a .lbl suffix')

        if min_longitude >= max_longitude:
            raise Exception('max_longitude must be larger than min_longitude')
        
        if min_latitude >= max_latitude:
            raise Exception('max_latitude must be larger than min_latitude')

        label = pvl.load(lbl_file_path)

        south_bound = int(label['IMAGE_MAP_PROJECTION']['MINIMUM_LATITUDE'])
        north_bound = int(label['IMAGE_MAP_PROJECTION']['MAXIMUM_LATITUDE'])

        east_bound = int(label['IMAGE_MAP_PROJECTION']['EASTERNMOST_LONGITUDE'])
        if east_bound > 180:
            east_bound = east_bound-360

        west_bound = int(label['IMAGE_MAP_PROJECTION']['WESTERNMOST_LONGITUDE'])
        if west_bound > 180:
            west_bound = west_bound-360

        if (min_longitude<west_bound or
            max_longitude>east_bound or
            min_latitude<south_bound or
            max_latitude>north_bound
        ):
            raise Exception("Latitude or longitude is out of bounds")

        
        rows = label['UNCOMPRESSED_FILE']['IMAGE']['LINES'] - 1
        cols = label['UNCOMPRESSED_FILE']['IMAGE']['LINE_SAMPLES'] -1

        min_col = int((min_longitude-west_bound) / (east_bound-west_bound) * cols)
        max_col = int((max_longitude-west_bound) / (east_bound-west_bound) * cols)
        min_row = int((max_latitude-north_bound) / (south_bound-north_bound) * rows)
        max_row = int((min_latitude-north_bound) / (south_bound-north_bound) * rows)

        window = Window(min_col, min_row, (max_col-min_col), (max_row-min_row))

        self.elevation, self.x, self.y = read_data(lbl_file_path, window)
        
        self.min_longitude = min_longitude
        self.max_longitude = max_longitude
        self.min_latitude = min_latitude
        self.max_latitude = max_latitude

    def long_lat_to_xy(self, long:float, lat:float)->np.ndarray:
        """
        Converts latitude and longitude to x and y coordinate points
    
        Parameters
        ----------
        long : float
            The longitude
        lat : float
            The latitude

        Returns
        -------
        coordinates : numpy.ndarray
            1D array in the form of [x, y].
        """

        x = (long - self.min_longitude)/(self.max_longitude-self.min_longitude) * self.x[-1]
        y = (lat - self.max_latitude)/(self.min_latitude-self.max_latitude) * self.y[-1]

        if(np.any(x<0) or np.any(x>self.x[-1]) or np.any(y<0) or np.any(y>self.y[-1])):
            raise Exception("Latitude or longitude is out of bounds")
            
        return np.array([x, y])

def read_data(lbl_file_path:Path, window:Window)->tuple:
    """
    Converts latitude and longitude to x and y coordinate points

    Parameters
    ----------
    lbl_file_path : pathlib.Path
        Path for the .lbl file of the data
    window : rasterio.window.Window
        Rasterio window object

    Returns
    -------
    data : numpy.ndarray
        The data inside of the .img file.
    x : numpy.ndarray
        1D array containing the coordinates in the west-east direction in meters. 
        0 at the westernmost point in the viewing window.
    y : numpy.ndarray
        1D array containing the coordinates in the north-south direction in meters.
        0 at the northenmost point in the viewing window.
    """
    label = pvl.load(lbl_file_path)
    with rasterio.open(lbl_file_path) as file:
        data = file.read(1, window=window) * label['UNCOMPRESSED_FILE']['IMAGE']['SCALING_FACTOR'] * 1000 # get elevation model in meters
        x_lin = np.arange(data.shape[1])*float(label['IMAGE_MAP_PROJECTION']['MAP_SCALE'])*1000
        y_lin = np.arange(data.shape[0])*float(label['IMAGE_MAP_PROJECTION']['MAP_SCALE'])*1000

    return data, x_lin, y_lin

def downsample(arr:np.ndarray, n:int)->np.ndarray:
    """
    Downsample a 2D array by averaging non-overlapping n x n blocks.
    
    Parameters
    ----------
    arr : numpy.ndarray
        2D array to be downscaled.
    n : int
        The downsampling factor.

    Returns
    -------
    result : np.ndarray
        The downsampled array.
    """
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
    )->tuple:
    """
    Finds the directional gradient tangent and normal to the provided direction 
    
    Parameters
    ----------
    pos : numpy.ndarray
        1D array of shape (2,) or 2D array of shape (N,2) containing the position to get the gradient.
    dir : numpy.ndarray
        1D array of shape (2,) or 2D array of shape (N,2) containing the unit direction vector for each point.
    x_grad_interp : scipy.interpolate.RegularGridInterpolator 
        A RegularGridInterpolator that returns the gradient in the x direction.
    y_grad_interp : scipy.interpolate.RegularGridInterpolator
        A RegularGridInterpolator that returns the gradient in the y direction.

    Returns
    -------
    forward : float or np.ndarray
        float or 1D array of shape (N,) containing the directional gradient in the tangent direction.
    sideways : float or np.ndarray
        float or 1D array of shape (N,) containing the directional gradient in the normal direction. 
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

def get_index(query:float, vec:np.ndarray)->int:
    """
    Finds the index of the closest point to the query point.
    
    Parameters
    ----------
    query : float
        The query point.
    vec : numpy.ndarray
        A contunially increasing 1D array.

    Returns
    -------
    result : int
        Returns the index of the item in vec that is the closest to query
    """
    i = np.arange(vec.size)
    return int(np.rint(np.interp(query, vec, i)))