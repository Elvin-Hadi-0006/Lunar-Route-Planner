import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import sys
from pathlib import Path
from time import time

parent_dir = Path(__file__).resolve().parent

sys.path.insert(0, str(parent_dir / "src"))

import main
import utils
plt.rcParams["figure.constrained_layout.use"] = True

lbl_path = parent_dir / "data" / "sldem2015_512_00n_30n_315_360_float.lbl"

"""
A couple of examples 

data = utils.SLDEM_reader(lbl_path, -8, 0, 13, 20)
start = data.long_lat_to_xy(-0.8, 14)
goal = data.long_lat_to_xy(-7, 19)

data = utils.SLDEM_reader(lbl_path, -29, -27.5, 14.2, 16.2)
start = data.long_lat_to_xy(-28.35, 16)
goal = data.long_lat_to_xy(-28.2, 14.5)
"""
data = utils.SLDEM_reader(lbl_path, -14, -7, 11, 18)
start = data.long_lat_to_xy(-13, 13)
goal = data.long_lat_to_xy(-8, 17)


weights = main.CostWeights(
    base_cost=10,
    uphill_cost=50,
    side_gradient_cost=30,
    downhill_cost=-10,
    relief_cost=0.1,
    roughness_cost=0.2,
    window_size=3
)
constraints = main.make_constraints_vector(
    uphill_max=0.2,
    side_gradient_max=0.3,
    downhill_max=0.4,
    relief_max=50,
    roughness_max=30
)

fig = plt.figure()

plt.imshow(data.elevation, extent=(data.x[0], data.x[-1], data.y[-1], data.y[0]), cmap='gray')
plt.colorbar(label="Elevation (m)")

solver = main.RoutePlanner(data.elevation, data.x, data.y)

start_time = time()

result = solver.find_route(
            start=start, 
            goal=goal,
            grid_factor=4,
            heuristics_weight=25,
            cost_weights=weights,
            constraints=constraints,
            line_of_sight_optimization=True,
            minimizer_optimization=False
            )

print(f"Runtime = {time() - start_time} s")

if result[2]:
    waypoints = result[0]
    plt.plot(waypoints[:,0], waypoints[:,1], 'b-')

else:
    visited_points = result[0]
    print(visited_points)
    plt.plot(visited_points[:,0], visited_points[:,1], 's', alpha=0.2)

plt.plot(start[0], start[-1], 'go', label='Start')
plt.plot(goal[0], goal[-1], 'ro', label='Goal')
fig.legend(loc='outside lower center', ncols=2)

plt.show()
