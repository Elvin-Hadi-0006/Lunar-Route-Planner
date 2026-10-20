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

map_path = Path(__file__).resolve().parent / "test_track.png"
syntethic_map = Image.open(map_path)
syntethic_map = ImageOps.grayscale(syntethic_map)
elevation = np.array(syntethic_map)

resolution = 1
start = np.array([35, 15])
end = np.array([105, 105])
x_lin = np.arange(0, elevation.shape[0]) * resolution
y_lin = np.arange(0, elevation.shape[1]) * resolution

weights = main.CostWeights(
    base_cost=1,
    uphill_cost=2,
    side_gradient_cost=1,
    downhill_cost=0,
    relief_cost=1,
    roughness_cost=1,
    window_size=3
)
constraints = main.make_constraints_vector(
    uphill_max=1,
    side_gradient_max=1,
    downhill_max=1,
    relief_max=10,
    roughness_max=100
)

solver = main.RoutePlanner(elevation, x_lin, y_lin)
result = solver.find_route(
    start=start, 
    goal=end,
    grid_factor=3,
    heuristics_weight=1,
    cost_weights=weights,
    constraints=constraints,
    line_of_sight_optimization=True,
    minimizer_optimization=True,
    minimizer_bounds_size=2,
    minimizer_max_iter=100,
    minimizer_constraint_tol=1e-4,
    minimizer_waypoints_tol=1e-1
    )

if result[2]:
    waypoints = result[0]
    cost = result[1]

    plt.imshow(elevation, extent=(x_lin[0], x_lin[-1], y_lin[-1], y_lin[0]))
    plt.plot(waypoints[:,1], waypoints[:,0])
    plt.plot(start[1], start[0], 'ro')
    plt.plot(end[1], end[0], 'ro')
    plt.show()
