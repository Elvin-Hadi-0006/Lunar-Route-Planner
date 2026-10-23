"""
Test using SLDEM lunar elevation data,
Tests the effect of grid_factor on the result
"""

import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import sys
from pathlib import Path
from time import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import main
import utils

plt.rcParams["figure.constrained_layout.use"] = True

lbl_path = Path(__file__).resolve().parent.parent / "data" / "sldem2015_512_00n_30n_315_360_float.lbl"

data = utils.SLDEM_reader(lbl_path, -29, -27.5, 14.2, 16.2)

start = data.long_lat_to_xy(-28.35, 16)
goal = data.long_lat_to_xy(-28.2, 14.5)

fig, ax = plt.subplots(1,3)
fig.set_size_inches(8,6)
ax[0].imshow(data.elevation, extent=(data.x[0], data.x[-1], data.y[-1], data.y[0]), cmap='gray')

solver = main.RoutePlanner(data.elevation, data.x, data.y)

weights = main.CostWeights(
    base_cost=10,
    uphill_cost=20,
    side_gradient_cost=10,
    downhill_cost=-5,
    relief_cost=0.03,
    roughness_cost=0.05,
    window_size=3
)
constraints = main.make_constraints_vector(
    uphill_max=0.3,
    side_gradient_max=0.4,
    downhill_max=0.5,
    relief_max=90,
    roughness_max=45
)

grid_factors = np.array([64])
final_costs = np.zeros(grid_factors.shape)
runtime = np.zeros(final_costs.shape)

for i in range(grid_factors.size): 
    print('\n--------------------------------------------')
    print(f"Grid factor = {grid_factors[i]}")

    start_time = time()

    result = solver.find_route(
                start=start, 
                goal=goal,
                grid_factor=grid_factors[i],
                heuristics_weight=30,
                cost_weights=weights,
                constraints=constraints,
                line_of_sight_optimization=True,
                minimizer_optimization=False
                )

    runtime[i] = time() - start_time

    if result[2]:
        final_costs[i] = result[1]
        waypoints = result[0]
        ax[0].plot(waypoints[:,0], waypoints[:,1], label=f'grid_factor = {grid_factors[i]}')

    else:
        visited_points = result[0]
        print(visited_points)
        ax[0].plot(visited_points[:,0], visited_points[:,1], 's', alpha=0.2, label=f'grid_factors = {grid_factors[i]}')

ax[0].plot(start[0], start[-1], 'ro', label='Start')
ax[0].plot(goal[0], goal[-1], 'ko', label='Goal')
fig.legend(loc='outside lower center', mode='expand', ncols=3)

bar_labels = [str(x) for x in grid_factors]

ax[1].bar(bar_labels, final_costs)
ax[1].set_xlabel('Grid factor')
ax[1].set_ylabel('Cost')
ax[1].grid('True', axis='y',which='both')

ax[2].bar(bar_labels, runtime)
ax[2].set_yscale('log', base=10)
ax[2].set_xlabel('Grid factor')
ax[2].set_ylabel('Runtime (s)')
ax[2].grid('True', axis='y',which='both')

fig.set_size_inches(6,4)
results_dir = Path(__file__).resolve().parent/'results'
#fig.savefig(results_dir/'lunar_data_grid_factor_test.png', dpi=300, bbox_inches='tight')

plt.show()
