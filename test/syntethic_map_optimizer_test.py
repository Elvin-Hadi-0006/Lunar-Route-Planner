"""
Test done on test_track.png
Tests the effect of grid_factor, line of sight optimization, and minimizer optimization on the result
"""

import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from matplotlib import pyplot as plt
import sys
from pathlib import Path
from time import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import main
plt.rcParams["figure.constrained_layout.use"] = True

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

# Elevation is transposed because PIL has a different format
solver = main.RoutePlanner(elevation.T, x_lin, y_lin)

fig, ax = plt.subplots(2,  3)
ax[0,0].set_title("(A) No optimization")
ax[0,1].set_title("(B) LOS")
ax[0,2].set_title("(C) LOS + minimizer")

grid_factors = np.array([1,3,5])
final_costs = np.zeros((3, grid_factors.size))
n_waypoints = np.zeros(final_costs.shape)
runtime = np.zeros(final_costs.shape)

for i in range(3):
    ax[0,i].imshow(elevation, extent=(x_lin[0], x_lin[-1], y_lin[-1], y_lin[0]))
    ax[0,i].plot(start[1], start[0], 'ro')
    ax[0,i].plot(end[1], end[0], 'ro')
    ax[0,i].axis('off')
                    

for i in range(grid_factors.size):
    grid_factor = grid_factors[i]

    for j in range(3):
        LOS_optimization = False
        minimizer_optimization = False
        if j == 1 or j==2:
            LOS_optimization = True
        if j==2:
            minimizer_optimization = True
        
        print("\n-----------------------------------------")
        print(f"Grid factor = {grid_factor}")
        print(f"Line of sight optimization = {LOS_optimization}")
        print(f"Minimizer optimization = {minimizer_optimization}\n")

        start_time = time()

        result = solver.find_route(
            start=start, 
            goal=end,
            grid_factor=grid_factor,
            heuristics_weight=1,
            cost_weights=weights,
            constraints=constraints,
            line_of_sight_optimization=LOS_optimization,
            minimizer_optimization=minimizer_optimization,
            minimizer_bounds_size=2,
            minimizer_max_iter=500,
            minimizer_constraint_tol=1e-4,
            minimizer_waypoints_tol=1e-1
            )

        runtime[j,i] = time() - start_time

        if result[2]:
            waypoints = result[0]
            ax[0,j].plot(waypoints[:,1], waypoints[:,0])
            
            final_costs[j,i] = result[1]
            n_waypoints[j,i] = waypoints.shape[0]


categories = ['A', 'B', 'C']
labels = [f"Grid factor = {n}" for n in grid_factors]
ax[1,0].grouped_bar(final_costs, tick_labels=categories, labels=labels)
ax[1,0].set_ylabel('Cost')

ax[1,1].grouped_bar(n_waypoints, tick_labels=categories)
ax[1,1].set_ylabel('Number of waypoints')
ax[1,1].set_yscale('log', base=10)
ax[1,1].grid('True', axis='y',which='both')

ax[1,2].grouped_bar(runtime, tick_labels=categories)
ax[1,2].set_ylabel('Runtime (s)')
ax[1,2].set_yscale('log', base=10)
ax[1,2].grid('True', axis='y',which='both')

fig.legend(loc="outside lower center")

fig.set_size_inches(6,4)
results_dir = Path(__file__).resolve().parent/'results'
fig.savefig(results_dir/'syntethic_map_optimizer_test.png', dpi=300, bbox_inches='tight')

plt.show()
