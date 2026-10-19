import sys
sys.path.insert(0,'/src')
import main 
import numpy as np
from scipy import integrate

weights = main.CostWeights(
    base_cost=1,
    uphill_cost=2,
    side_gradient_cost=1,
    downhill_cost=0,
    relief_cost=1,
    roughness_cost=1,
    window_size=3
)

elevation = np.ones((10,10))
x = np.arange(10)
y = np.arange(10)

solver = main.RoutePlanner(elevation, x, y)
main.evaluate_segment(0,0)