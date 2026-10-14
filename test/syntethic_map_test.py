import numpy as np
from scipy.interpolate import RegularGridInterpolator
from PIL import Image, ImageOps
from pathlib import Path
from matplotlib import pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import main
import utils

map_path = Path(__file__).resolve().parent / "test_track.jpg"
syntethic_map = Image.open(map_path)
syntethic_map = ImageOps.grayscale(syntethic_map)
elevation = np.array(syntethic_map)

resolution = 1
start = np.array([35, 15])
end = np.array([105, 105])
x_lin = np.arange(0, elevation.shape[0]) * resolution
y_lin = np.arange(0, elevation.shape[1]) * resolution
#x,y = np.meshgrid(x_lin, y_lin)

x_grad, y_grad = np.gradient(elevation, x_lin, y_lin)
x_grad_interp = RegularGridInterpolator((x_lin,y_lin), x_grad, bounds_error=False, fill_value=np.nan) 
y_grad_interp = RegularGridInterpolator((x_lin,y_lin), y_grad, bounds_error=False, fill_value=np.nan) 

uphill_cost = 2
sideways_cost = 1
downhill_cost = 0
base_cost = 1

uphill_constraint = 1
sideways_constraint = 1
downhill_constraint = 1

def eval_function(point:np.ndarray, dir:np.ndarray):
    forward_grad, sideways_grad = utils.directional_gradient(point, dir, x_grad_interp, y_grad_interp)
    
    cost = (np.where(forward_grad>=1, uphill_cost, downhill_cost) * forward_grad
            + sideways_cost * np.abs(sideways_grad) 
            + base_cost)

    constrained_vals = np.stack([forward_grad, sideways_grad, -forward_grad], axis=-1)

    return cost, constrained_vals

constraints = np.array([uphill_constraint, sideways_constraint, downhill_constraint])

result = main.a_star_search(start, end, x_lin, y_lin, eval_function, constraints, 2)

if result is not False:
    waypoints, cost = result

    plt.imshow(elevation, extent=(x_lin[0], x_lin[-1], y_lin[-1], y_lin[0]))
    plt.plot(waypoints[:,1], waypoints[:,0])
    plt.plot(start[1], start[0], 'ro')
    plt.plot(end[1], end[0], 'ro')
    plt.show()
else:
    print("A* Failed")