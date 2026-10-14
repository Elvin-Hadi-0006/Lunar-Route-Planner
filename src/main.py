import utils
import numpy as np
from scipy.interpolate import RegularGridInterpolator 
from scipy import integrate
from scipy import optimize
import rasterio
from rasterio.windows import Window
from pathlib import Path
import pvl
import heapq
from matplotlib import pyplot as plt

# TODO : Fix minimization step
class RoutePlanner:
    def __init__(self, start:np.ndarray, end:np.ndarray, n:int, func_eval, constraints_upper:np.ndarray, resolution, x_bounds:np.ndarray, y_bounds:np.ndarray):
        """
        func_eval(points, dir) : callable, returns (cost, constraints). 
        """
        self.start = start
        self.end = end
        self.n = n
        self.func_eval = func_eval
        self.constraints_upper = constraints_upper
        self.constraints = resolution
        self.resolution = resolution
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.last_waypoints_flat = None
        self.last_eval = None

    def update_and_evaluate_path(self, waypoints_flat:np.ndarray):
        # Only calculate if waypoints have changed
        if self.last_waypoints_flat is None or not np.array_equal(waypoints_flat, self.last_waypoints_flat):
            self.last_waypoints_flat = waypoints_flat
            waypoints = self.unflattened_waypoints(waypoints_flat)

            total_cost = 0
            constrained_vals = np.zeros(self.constraints_upper.shape)

            # Evaluate each segment of the path
            for i in range(self.n-1):
                # Create points inside the segment for calculations
                segmentStart = waypoints[i, :]
                segmentEnd = waypoints[i+1, :]
                segmentLength = np.sqrt(np.sum((segmentEnd - segmentStart)**2)) 
                pointsAmount = int(segmentLength/self.resolution) + 1
                points = np.stack(  [np.linspace(segmentStart[0], segmentEnd[0], pointsAmount),
                                    np.linspace(segmentStart[1], segmentEnd[1], pointsAmount)],
                                    axis=-1)
                dir = np.repeat(np.atleast_2d((segmentEnd - segmentStart) / segmentLength), pointsAmount, 0)

                # Evaluates costs and constrained values
                segment_cost, segment_constrained_vals = self.func_eval(points, dir)
                total_cost += integrate.simpson(y=segment_cost, dx=segmentLength/pointsAmount)
                segment_constrained_vals_max = np.max(segment_constrained_vals, axis=1)
                constrained_vals = np.maximum(segment_constrained_vals_max, constrained_vals)
    
            self.last_eval = total_cost, constrained_vals 

        return self.last_eval

    def cost(self, waypoints_flat:np.ndarray=None):
        return self.update_and_evaluate_path(waypoints_flat)[0]

    def constrained_vals(self, waypoints_flat:np.ndarray=None):
        return self.update_and_evaluate_path(waypoints_flat)[1]

    def unflattened_waypoints(self, waypoinys_flat):
        return waypoinys_flat.reshape(self.n, 2)

    def solve(self, verbose=2):
        star_end_tolerance = self.resolution/2

        # Create Bounds object
        lower_bound = np.zeros((self.n,2))
        lower_bound[1:-2,0] = self.x_bounds[0]
        lower_bound[1:-2,1] = self.y_bounds[0]
        lower_bound[0,:] = self.start - star_end_tolerance
        lower_bound[-1,:] = self.end - star_end_tolerance
        lower_bound = lower_bound.flatten()

        upper_bound = np.zeros((self.n,2))
        upper_bound[1:-2,0] = self.x_bounds[1]
        upper_bound[1:-2,1] = self.y_bounds[1]
        upper_bound[0,:] = self.start + star_end_tolerance
        upper_bound[-1,:] = self.end + star_end_tolerance
        upper_bound = upper_bound.flatten()

        bounds = optimize.Bounds(lower_bound, upper_bound)

        # Create NonLinearConstraints object
        nonlinear_constraints = optimize.NonlinearConstraint(
            self.constrained_vals, 
            np.ones(self.constraints_upper.size)*(-np.inf), 
            self.constraints_upper
        )

        result = optimize.shgo(self.cost, bounds, 
                               constraints=nonlinear_constraints, 
                               minimizer_kwargs={'method':'trust-constr'}, 
                               options={'disp':True}, callback=print)

        return result


def a_star_search(start, goal, x, y, eval_func, constraints, heuristic_weight):
    start_i = utils.get_index(start[0], x)
    start_j = utils.get_index(start[1], y)
    goal_i = utils.get_index(goal[0], x)
    goal_j = utils.get_index(goal[1], y)

    open_set = []
    heapq.heapify(open_set)
    heapq.heappush(open_set, (0, start_i, start_j))
    closed_set = [(start_i, start_j)]

    g_score = np.ones((x.size, y.size)) * np.inf
    g_score[start_i, start_j] = 0
    came_from={(start_i, start_j):None}


    while open_set:
        _, current_i, current_j = heapq.heappop(open_set)

        current_x = x[current_i]
        current_y = y[current_j]

        # If goal has been reached, reconstruct path and return result
        if current_i == goal_i and current_j == goal_j:
            route = np.array([[current_x, current_y]])
            prev_node = came_from[(current_i, current_j)]
            while prev_node is not None:
                route = np.append(np.array([[x[prev_node[0]], y[prev_node[1]]]]), route, axis=0)
                prev_node = came_from[prev_node]

            return route, g_score[goal_i, goal_j]

        neighbour_i_list = np.array([-1, 0, 1, -1, 1, -1, 0, 1]) + current_i
        neighbour_j_list = np.array([-1, -1, -1, 0, 0, 1, 1, 1]) + current_j

        for k in range(8):
            # check if within bounds 
            neighbour_i = neighbour_i_list[k]
            neighbour_j = neighbour_j_list[k]

            if not (
                neighbour_i < 0 or
                neighbour_i >= x.size or
                neighbour_j < 0 or
                neighbour_j >= y.size
            ):
                neighbour_x = x[neighbour_i]
                neighbour_y = y[neighbour_j]
                
                dist_to_neighbour = np.sqrt((neighbour_x-current_x)**2 + (neighbour_y-current_y)**2)
                midpoint_xy = np.array([neighbour_x+current_x, neighbour_y+current_y])/2
                dir_to_neighbour = np.array([neighbour_x-current_x, neighbour_y-current_y]) / dist_to_neighbour

                midpoint_cost, midpoint_constraints = eval_func(midpoint_xy, dir_to_neighbour)

                # Check there is a valid route to the neighbour (aproximated by checking the midpoint) 
                if np.all(midpoint_constraints < constraints):

                    neighbour_g_score = midpoint_cost*dist_to_neighbour + g_score[current_i, current_j]

                    if(neighbour_g_score.item() < g_score[neighbour_i, neighbour_j]):
                        g_score[neighbour_i, neighbour_j] = neighbour_g_score.item()
                        
                        dist_to_goal = np.sqrt((goal[0]-neighbour_x)**2 + (goal[1]-neighbour_y)**2)
                        neighbour_f_score = neighbour_g_score + heuristic_weight * dist_to_goal

                        came_from[(neighbour_i, neighbour_j)]=(current_i, current_j)

                        if (neighbour_i, neighbour_j) not in closed_set:
                            heapq.heappush(open_set, (neighbour_f_score, neighbour_i, neighbour_j))
                            closed_set.append((neighbour_i, neighbour_j))                                            

    plt.imshow(g_score)
    plt.show()

    return False