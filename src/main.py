import utils
import numpy as np
from scipy.interpolate import RegularGridInterpolator 
from scipy import integrate, optimize, ndimage
import rasterio
from rasterio.windows import Window
from pathlib import Path
import pvl
import heapq
from matplotlib import pyplot as plt

def evaluate_segment(segment_start, segment_end, resolution, func_eval):
    # Create points inside the segment for calculations
    segment_length = np.sqrt(np.sum((segment_end - segment_start)**2)) 
    points_amount = int(segment_length/resolution) + 1
    points = np.stack(  [np.linspace(segment_start[0], segment_end[0], points_amount),
                        np.linspace(segment_start[1], segment_end[1], points_amount)],
                        axis=-1)
    dir = np.repeat(np.atleast_2d((segment_end - segment_start) / segment_length), points_amount, 0)

    # Evaluates costs and constrained values
    point_cost, point_constrained_vals = func_eval(points, dir)
    segment_cost = integrate.trapezoid(y=point_cost, dx=segment_length/points_amount)
    segment_constrained_vals = np.max(point_constrained_vals, axis=0)

    return segment_cost, segment_constrained_vals

# TODO : Fix minimization step
class MinimizeRouteCost:
    def __init__(self, 
                 start:np.ndarray, 
                 end:np.ndarray, 
                 initial_waypoints:np.ndarray, 
                 func_eval, 
                 constraints:np.ndarray, 
                 resolution, 
                 bounds_size 
                 ):
        """
        func_eval(points, dir) : callable, returns (cost, constraints). 
        """
        self.start = start
        self.end = end
        self.initial_waypoints = initial_waypoints
        self.func_eval = func_eval
        self.constraints = constraints
        self.resolution = resolution
        self.bounds_size = bounds_size
        self.n = initial_waypoints.shape[0]
        self.last_waypoints_flat = None
        self.last_eval = None

    def update_and_evaluate_path(self, waypoints_flat:np.ndarray):
        # Only calculate if waypoints have changed
        if self.last_waypoints_flat is None or not np.array_equal(waypoints_flat, self.last_waypoints_flat):
            self.last_waypoints_flat = waypoints_flat
            waypoints = self.unflattened_waypoints(waypoints_flat)

            total_cost = 0
            constrained_vals = np.zeros(self.constraints.shape)

            # Evaluate each segment of the path
            for i in range(self.n-1):
                segment_start = waypoints[i, :]
                segment_end = waypoints[i+1, :]
                segment_cost, segment_constrained_vals = evaluate_segment(segment_start, segment_end, self.resolution, self.func_eval)
                total_cost +=  segment_cost
                constrained_vals = np.maximum(segment_constrained_vals,constrained_vals)
    
            self.last_eval = total_cost, constrained_vals 

        return self.last_eval

    def cost(self, waypoints_flat:np.ndarray=None):
        return self.update_and_evaluate_path(waypoints_flat)[0]

    def constrained_vals(self, waypoints_flat:np.ndarray=None):
        return self.update_and_evaluate_path(waypoints_flat)[1]

    def unflattened_waypoints(self, waypoinys_flat):
        return waypoinys_flat.reshape(self.n, 2)

    def solve(self, verbose=2)->optimize.OptimizeResult:
        initial_waypoints_flat = self.initial_waypoints.flatten()

        # Create Bounds object
        lower_bound = initial_waypoints_flat
        lower_bound[0] -= self.resolution/2
        lower_bound[-1] -= self.resolution/2
        lower_bound[1:-2] -= self.bounds_size

        upper_bound = initial_waypoints_flat
        upper_bound[0] += self.resolution/2
        upper_bound[-1] += self.resolution/2
        upper_bound[1:-2] += self.bounds_size

        bounds = optimize.Bounds(lower_bound, upper_bound)

        # Create NonLinearConstraints object
        nonlinear_constraints = optimize.NonlinearConstraint(
            self.constrained_vals, 
            np.ones(self.constraints.shape)*(-np.inf), 
            self.constraints
        )

        result = optimize.minimize(
            fun=self.cost,
            x0=initial_waypoints_flat,
            method='trust-constr',
            bounds=bounds,
            constraints=nonlinear_constraints,
            options={'verbose':2},
            tol=self.resolution/10
        )
        return result


def a_star_search(start, goal, x, y, eval_func, eval_resolution, constraints, heuristic_weight):
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

            return route, g_score[goal_i, goal_j], True

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

                segment_cost, segment_constrained_vals = evaluate_segment(
                    segment_start=np.array([current_x, current_y]),
                    segment_end=np.array([neighbour_x, neighbour_y]),
                    resolution=eval_resolution,
                    func_eval=eval_func
                )

                # Check there is a valid route to the neighbour 
                if np.all(segment_constrained_vals < constraints):

                    neighbour_g_score = segment_cost + g_score[current_i, current_j]

                    if(neighbour_g_score.item() < g_score[neighbour_i, neighbour_j]):
                        g_score[neighbour_i, neighbour_j] = neighbour_g_score.item()
                        
                        dist_to_goal = np.sqrt((goal[0]-neighbour_x)**2 + (goal[1]-neighbour_y)**2)
                        neighbour_f_score = neighbour_g_score + heuristic_weight * dist_to_goal

                        came_from[(neighbour_i, neighbour_j)]=(current_i, current_j)

                        if (neighbour_i, neighbour_j) not in closed_set:
                            heapq.heappush(open_set, (neighbour_f_score, neighbour_i, neighbour_j))
                            closed_set.append((neighbour_i, neighbour_j))                                            

    return np.array([]), np.array([]), False

class CostWeights:
    def __init__(self, base_cost, uphill_cost, side_gradient_cost, downhill_cost, relief_cost, roughness_cost):
        self.base_cost = base_cost
        self.uphill_cost = uphill_cost
        self.side_gradient_cost = side_gradient_cost
        self.downhill_cost = downhill_cost
        self.relief_cost = relief_cost
        self.roughness_cost = roughness_cost

def make_constraints_vector(uphill_max, side_gradient_max, downhill_max, relief_max, roughness_max):
    return np.array([uphill_max,
                     side_gradient_max,
                     downhill_max,
                     relief_max,
                     roughness_max])

class RoutePlanner:
    def __init__(self, elevation, x, y):
        self.elevation = elevation
        self.x = x
        self.y = y
        self.resolution = np.abs(x[1]-x[0])

    def eval_function(self, 
                      point:np.ndarray, 
                      dir:np.ndarray,
                      x_grad_interp:RegularGridInterpolator,
                      y_grad_interp:RegularGridInterpolator,
                      relief_interp:RegularGridInterpolator,
                      roughness_interp:RegularGridInterpolator,
                      weights:CostWeights
                      ): 
        forward_grad, sideways_grad = utils.directional_gradient(point, dir, x_grad_interp, y_grad_interp)
        relief = relief_interp(point)
        roughness = roughness_interp(point)

        cost = (weights.base_cost
                + np.where(forward_grad>0, weights.uphill_cost, -weights.downhill_cost) * forward_grad
                + weights.side_gradient_cost * np.abs(sideways_grad) 
                + weights.relief_cost * relief
                + weights.roughness_cost * roughness
                )

        constrained_vals = np.stack(np.atleast_2d([forward_grad, sideways_grad, -forward_grad, relief, roughness]), axis=-1)

        return cost, constrained_vals

    def find_route(self,
                   start:np.ndarray, 
                   goal:np.ndarray, 
                   coarse_factor:int,
                   coarse_heuristics_weight,
                   coarse_weights:CostWeights, 
                   coarse_constraints:np.ndarray,
                   optimizer_bounds_size,
                   fine_weights:CostWeights,
                   fine_constraints:np.ndarray,
                   optimize_route:bool=True
                ):

        x_grad, y_grad = np.gradient(self.elevation, self.x, self.y)
        x_grad_interp = RegularGridInterpolator((self.x,self.y), x_grad, bounds_error=False, fill_value=np.nan) 
        y_grad_interp = RegularGridInterpolator((self.x,self.y), y_grad, bounds_error=False, fill_value=np.nan) 

        coarse_relief = ndimage.maximum_filter(self.elevation, coarse_factor) - ndimage.minimum_filter(self.elevation, coarse_factor)
        coase_relief_interp = RegularGridInterpolator((self.x,self.y), coarse_relief, bounds_error=False, fill_value=np.nan)

        coarse_roughness = np.sqrt(
            ndimage.uniform_filter(
                np.square(
                    self.elevation - ndimage.uniform_filter(self.elevation, size=coarse_factor)
                )
            , size=coarse_factor)
        )
        coarse_roughness_interp = RegularGridInterpolator((self.x,self.y), coarse_roughness, bounds_error=False, fill_value=np.nan)

        print("Using A* search on coarse grid")

        coarse_eval_func = lambda point, dir: self.eval_function(
            point=point,
            dir=dir,
            x_grad_interp=x_grad_interp,
            y_grad_interp=y_grad_interp,
            relief_interp=coase_relief_interp,
            roughness_interp=coarse_roughness_interp,
            weights=coarse_weights
        )

        coarse_route, coarse_cost, a_star_success = a_star_search(
            start=start,
            goal=goal,
            x=self.x[int(coarse_factor/2)::coarse_factor],
            y=self.y[int(coarse_factor/2)::coarse_factor],
            eval_func=coarse_eval_func,
            eval_resolution=self.resolution,
            constraints=coarse_constraints,
            heuristic_weight=coarse_heuristics_weight
            )

        if not a_star_success:
            print("A* search failed to find goal")
            return None, None, False, False

        print("A* search completed")

        fine_relief = ndimage.maximum_filter(self.elevation, 3) - ndimage.minimum_filter(self.elevation, 3)
        fine_relief_interp = RegularGridInterpolator((self.x,self.y), fine_relief, bounds_error=False, fill_value=np.nan)

        fine_roughness = np.sqrt(
            ndimage.uniform_filter(
                np.square(
                    self.elevation - ndimage.uniform_filter(self.elevation, size=3)
                )
            , size=3)
        )
        fine_roughness_interp = RegularGridInterpolator((self.x,self.y), fine_roughness, bounds_error=False, fill_value=np.nan)

        if not optimize_route:
            return coarse_route, coarse_cost, True, False

        print("Optimizing coarse route")

        fine_eval_func = lambda point, dir: self.eval_function(
                    point=point,
                    dir=dir,
                    x_grad_interp=x_grad_interp,
                    y_grad_interp=y_grad_interp,
                    relief_interp=fine_relief_interp,
                    roughness_interp=fine_roughness_interp,
                    weights=fine_weights
                )
        
        route_optimizer = MinimizeRouteCost(
            start=start,
            end=goal,
            initial_waypoints=coarse_route,
            func_eval=fine_eval_func,
            constraints=fine_constraints,
            resolution=self.resolution,
            bounds_size=optimizer_bounds_size
        )

        optimizer_result = route_optimizer.solve()

        if not optimizer_result.success:
            print("Minimizer failed to converge")
            print(optimizer_result.message)
            return coarse_route, coarse_cost, True, False

        print("Optimization completed")

        return (route_optimizer.unflattened_waypoints(optimizer_result.x),
            optimizer_result.fun,
            True,
            True
        )