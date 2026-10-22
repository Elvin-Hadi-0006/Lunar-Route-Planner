"""
Test done on test_track.png
Tests the effect of the heuristics_weight on the result
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
import utils

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

solver = main.RoutePlanner(elevation.T, x_lin, y_lin)

fig, ax = plt.subplots(2, 2)

heuristic_weights = np.array([0, 1, 3, 5, 10])
final_costs = np.zeros((2, heuristic_weights.size))
runtime = np.zeros(final_costs.shape)

ax[0,0].set_title('(A) No optimisation')
ax[0,1].set_title('(B) Line of sight')

for i in range(2):
    ax[0,i].imshow(elevation, extent=(x_lin[0], x_lin[-1], y_lin[-1], y_lin[0]))
    ax[0,i].plot(start[1], start[0], 'ro')
    ax[0,i].plot(end[1], end[0], 'ro')
    ax[0,i].axis('off')
                    
for i in [0,1]:
    LOS_optimization = bool(i)

    for j in range(heuristic_weights.size):
        heuristic_weight = heuristic_weights[j]
        
        print("\n-----------------------------------------")
        print(f"Heuristic weight = {heuristic_weight}")
        print(f"Line of sight optimization = {LOS_optimization}")

        start_time = time()

        result = solver.find_route(
            start=start, 
            goal=end,
            grid_factor=1,
            heuristics_weight=heuristic_weight,
            cost_weights=weights,
            constraints=constraints,
            line_of_sight_optimization=LOS_optimization,
            minimizer_optimization=False
            )

        runtime[i,j] = time() - start_time

        if result[2]:
            waypoints = result[0]
            ax[0,i].plot(waypoints[:,1], waypoints[:,0])
            
            final_costs[i,j] = result[1]


categories = ['A', 'B']
labels = [f"Heuristic weight = {n}" for n in heuristic_weights]
ax[1,0].set_title("Final cost")
ax[1,0].grouped_bar(final_costs, tick_labels=categories, labels=labels)
ax[1,0].set_ylabel('Cost')
ax[1,1].set_title("Runtime")
ax[1,1].grouped_bar(runtime, tick_labels=categories)
ax[1,1].set_ylabel('Runtime (s)')

fig.legend(loc="outside right")

fig.set_size_inches(6,4)
results_dir = Path(__file__).resolve().parent/'results'
fig.savefig(results_dir/'syntethic_map_heuristic_test.png', dpi=300, bbox_inches='tight')

plt.show()
