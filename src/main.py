import utils
import numpy as np
from scipy.interpolate import RegularGridInterpolator 
from scipy import integrate, optimize, ndimage
import heapq

def evaluate_segment(
        segment_start:np.ndarray, 
        segment_end:np.ndarray, 
        eval_resolution:float, 
        eval_func:callable
        )->tuple:
    """
    Evaluates the cost and constrained variables between 2 points

    Parameters
    ----------
    segment_start : numpy.ndarray
        1D array of shape (2, ) containting the point where the segment starts.
    segment_end : numpy.ndarray
        1D array of shape (2, ) containting the point where the segment ends.
    eval_resolution : float
        The distance between each point when integrating eval_func
    eval_func : callable
        A function that accepts (pos, dir) and retruns cost and constrained values. pos and dir are an ndarrays of shape (N,2).  

    Returns
    -------
    segment_cost : float
        The integrated cost through the segment
    segment_constrained_vals : numpy.ndarray
        1D ndarray of the maximmum constrained values through the segment    
    """

    # Create points inside the segment for calculations
    segment_length = np.sqrt(np.sum((segment_end - segment_start)**2)) 
    points_amount = int(segment_length/eval_resolution) + 1
    points_amount = np.maximum(points_amount, 2)
    points = np.stack(  [np.linspace(segment_start[0], segment_end[0], points_amount),
                        np.linspace(segment_start[1], segment_end[1], points_amount)],
                        axis=-1)
    dir = np.repeat(np.atleast_2d((segment_end - segment_start) / segment_length), points_amount, 0)

    # Evaluates costs and constrained values
    point_cost, point_constrained_vals = eval_func(points, dir)
    segment_cost = integrate.trapezoid(y=point_cost, dx=segment_length/(points_amount-1))
    segment_constrained_vals = np.max(point_constrained_vals, axis=0)

    return segment_cost, segment_constrained_vals

def evaluate_route(waypoints:np.ndarray, 
                   eval_resolution:float, 
                   eval_func:callable, 
                   n_constraints:int
                   )->tuple:
    """
    Evaluates the cost and constrained variables along a route consisting of many waypoints

    Parameters
    ----------
    waypoints : numpy.ndarray
        2D array of shape (2, N) containting waypoints making up the route.
    eval_resolution : float
        The distance between each point when integrating eval_func
    eval_func : callable
        A function that accepts (pos, dir) and retruns cost and constrained values. pos and dir are an ndarrays of shape (N,2).  
    n_constraints : int
        The number of constrained variables

    Returns
    -------
    total_cost : numpy.ndarray
        The total cost through the route
    constrained_vals : numpy.ndarray
        1D ndarray of the maximmum constrained values through the route    
    """
    
    total_cost = 0
    constrained_vals = np.zeros(n_constraints)

    # Add new points inbetween current waypoints
    dist_between_waypoints = np.sqrt(np.sum( (waypoints[1:,:]-waypoints[:-1,:])**2, axis=1 ))
    steps_per_gap = np.astype(dist_between_waypoints/eval_resolution, int) + 1

    new_index = np.arange(0, np.sum(steps_per_gap))
    old_index = np.append(0, np.cumsum(steps_per_gap))

    new_waypoints = np.vstack(
        (np.interp(new_index, old_index, waypoints[:,0]),
         np.interp(new_index, old_index, waypoints[:,1]))
    ).T

    delta_new_waypoints = new_waypoints[1:,:] - new_waypoints[:-1,:]

    dist_between_new_waypoints = np.sqrt(np.sum( delta_new_waypoints**2, axis=1 ))

    dir = delta_new_waypoints / dist_between_new_waypoints[:,np.newaxis]
    dir = np.append(dir, dir[-1:,:], axis=0)
    cumulative_distance = np.append(0,np.cumsum(dist_between_new_waypoints))

    # Evaluates costs and constrained values
    point_cost, point_constrained_vals = eval_func(new_waypoints, dir)
    total_cost = integrate.trapezoid(y=point_cost, x=cumulative_distance)
    constrained_vals = np.max(point_constrained_vals, axis=0)

    return total_cost, constrained_vals 

def a_star_search(start:np.ndarray, 
                  goal:np.ndarray, 
                  x:np.ndarray, 
                  y:np.ndarray, 
                  eval_func:callable, 
                  eval_resolution:float, 
                  constraints:np.ndarray, 
                  heuristic_weight:float, 
                  apply_line_of_sight_optimization:bool
                  )->tuple:
    """
    A* search algorithm with line of sight optimization

    Parameters
    ----------
    start : numpy.ndarray
        The position to start the search. 1D array of shape (2,) containing the x and y coordinates
    goal : numpy.ndarray
        The position of the goal. 1D array of shape (2,) containing the x and y coordinates
    x : numpy.ndarray
        A 1D array containing coordinate points in the x direction.
    y : numpy.ndarray
        A 1D array containing coordinate points in the y direction.
    eval_func : callable
        A function that accepts (pos, dir) and retruns cost and constrained values. pos and dir are an ndarrays of shape (N,2).  
    eval_resolution : float
        The distance between each point when integrating eval_func
    constraints : numpy.ndarray
        The maximum values for each constrained variables
    heuristic_weight : float
        The weight of the heuristic term h(x)
    apply_line_of_sight_optimization : bool
        Set to True to apply line of sight optimization
    
    Returns
    -------
    waypoints : numpy.ndarray
        ndarray of shape (N,2). Contains the waypoints making up the route if search is succesfull. 
        Contains all of the visited cells if unsuccessfull.
    cost : float
        The final cost of the route. None if search is unsuccessful.
    success_flag : bool
        True if the search is successfull.
    """

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
            waypoints = np.array([[current_x, current_y]])
            prev_node = came_from[(current_i, current_j)]
            
            print()
            
            if apply_line_of_sight_optimization:
                waypoints_g_score = np.array(g_score[current_i, current_j])
                while prev_node is not None:
                    waypoints = np.append(np.array([[x[prev_node[0]], y[prev_node[1]]]]), waypoints, axis=0)
                    waypoints_g_score = np.append(g_score[prev_node[0], prev_node[1]], waypoints_g_score)
                    prev_node = came_from[prev_node] 

                print(f"A* search completed with {waypoints_g_score[-1]:.4g} cost and {waypoints.size} waypoints")
                print("Applying line of sight optimization")

                waypoints, cost = line_of_sight_optimization(waypoints, waypoints_g_score, eval_func, eval_resolution, constraints)

            else:
                cost = g_score[current_i, current_j]
                while prev_node is not None:
                    waypoints = np.append(np.array([[x[prev_node[0]], y[prev_node[1]]]]), waypoints, axis=0)
                    prev_node = came_from[prev_node]

            print(f"A* search completed with {cost:.4g} cost and {waypoints.size} waypoints")
            return waypoints, cost, True


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
                    eval_resolution=eval_resolution,
                    eval_func=eval_func
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
        
        closed_set.append((current_i, current_j))      
        print(f"\rVisited cells = {len(closed_set)}", end='')                                      

    visited_x = np.zeros(len(closed_set))
    visited_y = np.zeros(len(closed_set))
    for i in range(len(closed_set)):
        visited_x[i] = x[closed_set[i][0]]
        visited_y[i] = y[closed_set[i][1]]

    return np.stack([visited_x, visited_y], axis=-1), None, False

def line_of_sight_optimization(waypoints:np.ndarray, 
                               waypoints_g_score:np.ndarray, 
                               eval_func:callable, 
                               eval_resolution:float, 
                               constraints:np.ndarray
                               )->tuple:
    """
    Line of sight optimization for A* search. Deletes redundant waypoints. 

    Parameters
    ----------
    waypoints : numpy.ndarray
        2D array of shape (2, N) containting waypoints making up the route
    goal : numpy.ndarray
        The position of the goal. 1D array of shape (2,) containing the x and y coordinates
    eval_func : callable
        A function that accepts (pos, dir) and retruns cost and constrained values. pos and dir are an ndarrays of shape (N,2).  
    eval_resolution : float
        The distance between each point when integrating eval_func
    constraints : numpy.ndarray
        The maximum values for each constrained variables
    
    Returns
    -------
    optimized_waypoints : numpy.ndarray
        2D array of shape (2, N) containting waypoints making up the optimized route
    optimized_cost : float
        Cost of the optimized route.
    """
    
    optimized_route = np.array([waypoints[0,:]])
    optimized_cost = 0
    current_index = 0
    next_index = 2
    n = waypoints.shape[0]
    prev_segment_cost = waypoints_g_score[1]
    
    while True:

        # If index is out of bounds, return the optimized result
        if(next_index == n):
            optimized_route = np.append(optimized_route, np.array([waypoints[next_index-1,:]]), axis=0)
            optimized_cost += prev_segment_cost
            return optimized_route, optimized_cost

        segment_cost, segment_constrained_vals = evaluate_segment(
            segment_start=waypoints[current_index,:],
            segment_end=waypoints[next_index,:],
            eval_resolution=eval_resolution,
            eval_func=eval_func
        )

        # print(f"{current_index}, {next_index}, \t {waypoints_g_score[next_index]-waypoints_g_score[current_index]}, \t {prev_segment_cost},\t {segment_cost}")

        if (np.any(segment_constrained_vals > constraints) or
            (segment_cost > waypoints_g_score[next_index]-waypoints_g_score[current_index]) 
        ):
            optimized_route = np.append(optimized_route, np.array([waypoints[next_index-1,:]]), axis=0)
            current_index = next_index-1
            optimized_cost += prev_segment_cost
            prev_segment_cost = waypoints_g_score[current_index+1]-waypoints_g_score[current_index]
        else:
            prev_segment_cost = segment_cost

        next_index += 1       

def make_constraints_vector(
        uphill_max, 
        side_gradient_max, 
        downhill_max, 
        relief_max, 
        roughness_max
        ):
    """
    Utility function for creating the constraints vector

    Parameters
    ----------
    uphill_max : float 
        Maximum uphill gradient 
    side_gradient_max : float 
        Maximum sideways gradient
    downhill_max : float
        Maximum downhill gradient
    relief_max : float
        Maximum terain relief (maximum change in elevation inside of an (n x n) window)
    roughness_max : float 
        Maximmum terain roughness (standard deviation of the elevation inside an (n x n) window)

    Returns
    ----------
    constraints_vector : numpy.ndarray
        The maximum constraints vector that can be passed to RoutePlanner.find_route()
    """
    return np.array([uphill_max,
                     side_gradient_max,
                     downhill_max,
                     relief_max,
                     roughness_max])

class CostWeights:
    """
    Class containing the weights for RoutePlanner.eval_function

    Parameters
    ----------
    base_cost : float
        The base cost
    uphill_cost : float 
        The cost of an uphill gradient 
    side_gradient_cost : float 
        The cost of an sideways gradient
    downhill_cost : float
        The cost of an downhill gradient
    relief_cost : float
        The cost of terain relief (maximum change in elevation inside of an (n x n) window)
    roughness_cost : float 
        The cost of terain roughness (standard deviation of the elevation inside an (n x n) window)
    window_size : int
        The size of the window to evaluate relief and roughness

    Attributes
    ----------
    base_cost : float
        The base cost
    uphill_cost : float 
        The cost of an uphill gradient 
    side_gradient_cost : float 
        The cost of an sideways gradient
    downhill_cost : float
        The cost of an downhill gradient
    relief_cost : float
        The cost of terain relief (maximum change in elevation inside of an (n x n) window)
    roughness_cost : float 
        The cost of terain roughness (standard deviation of the elevation inside an (n x n) window)
    window_size : int
        The size of the window to evaluate relief and roughness
    """

    def __init__(self, 
                 base_cost : float, 
                 uphill_cost : float, 
                 side_gradient_cost : float, 
                 downhill_cost : float, 
                 relief_cost : float, 
                 roughness_cost : float, 
                 window_size : int):
        self.base_cost = base_cost
        self.uphill_cost = uphill_cost
        self.side_gradient_cost = side_gradient_cost
        self.downhill_cost = downhill_cost
        self.relief_cost = relief_cost
        self.roughness_cost = roughness_cost
        self.window_size = window_size

class MinimizeRouteCost:
    """
    Minimize a route's cost by using the COBYQA minimization algorithm

    Parameters
    ----------
    initial_waypoints : np.ndarray
        2D array of shape (N,2) containing the waypoints of the route to minimize the cost of.
    eval_func : callable
        A function that accepts (pos, dir) and retruns cost and constrained values. pos and dir are an ndarrays of shape (N,2).  
    constraints : numpy.ndarray
        The maximum values for each constrained variables
    eval_resolution : float
        The distance between each point when integrating eval_func
    bounds_size:float 
        The size of the boundsfor the minimizer. 
        The minimizer will only evaluate points inside of an (n unit x n unit) window arround the initial waypoints.

    Attributes
    ----------
    initial_waypoints : np.ndarray
        2D array of shape (N,2) containing the waypoints of the route to minimize the cost of.
    eval_func : callable
        A function that accepts (pos, dir) and retruns cost and constrained values. pos and dir are an ndarrays of shape (N,2).  
    constraints : numpy.ndarray
        The maximum values for each constrained variables
    eval_resolution : float
        The distance between each point when integrating eval_func
    bounds_size : float 
        The size of the boundsfor the minimizer. 
        The minimizer will only evaluate points inside of an (n unit x n unit) window arround the initial waypoints.
    last_waypoints_flat : numpy.ndarray
        1D array containing the flattened waypoints that is the result of the previous call update_and_evaluate_path().
        This value is stored to reduce unnecesarry calls to the evaulate_route function.
    last_eval : tuple(float, numpy.ndarray)
        A tuple containing the cost and constrained values that is the result of the previous call of update_and_evaluate_path().
        These value are stored to reduce unnecesarry calls to the evaulate_route function.
    """
    def __init__(self, 
                 initial_waypoints:np.ndarray, 
                 eval_func:callable, 
                 constraints:np.ndarray, 
                 resolution:float, 
                 bounds_size:float 
                 ):
        self.initial_waypoints = initial_waypoints
        self.eval_func = eval_func
        self.constraints = constraints
        self.resolution = resolution
        self.bounds_size = bounds_size
        self.n = initial_waypoints.shape[0]
        self.last_waypoints_flat = None
        self.last_eval = None

    def update_and_evaluate_path(self, waypoints_flat:np.ndarray)->tuple:
        """
        Update and evalates the route. A wrapper for the evaluate_route function.
        Does not call the evaulate_routue function if the same parameters are given twice in a row.
        Instead, the results from the previous call is returned instead. 

        Parameters
        ----------
        waypoints_flat : numpy.ndarray
            1D array of containting waypoints making up the route that has been passed through numpy.flatten()
        
        Returns
        -------
        self.last_eval : tuple(float, numpy.ndarray)
        A tuple containing the cost and constrained values from the most recent call to the evaluate_route function.
        """
        # Only calculate if waypoints have changed
        if self.last_waypoints_flat is None or not np.array_equal(waypoints_flat, self.last_waypoints_flat):
            self.last_waypoints_flat = waypoints_flat
            waypoints = self.unflattened_waypoints(waypoints_flat)

            self.last_eval = evaluate_route(waypoints, self.resolution, self.eval_func, self.constraints.size)

        return self.last_eval

    def cost(self, waypoints_flat:np.ndarray=None)->float:
        """
        A wrapper for update_and_evaluate_path function that returns the cost.

        Parameters
        ----------
        waypoints_flat : numpy.ndarray
            1D array of containting waypoints making up the route that has been passed through numpy.flatten()
        
        Returns
        -------
        cost : float
        The cost of the route
        """
        return self.update_and_evaluate_path(waypoints_flat)[0]

    def constrained_vals(self, waypoints_flat:np.ndarray=None)->cost:
        """
        A wrapper for update_and_evaluate_path function that returns the constrained values.

        Parameters
        ----------
        waypoints_flat : numpy.ndarray
            1D array of containting waypoints making up the route that has been passed through numpy.flatten()
        
        Returns
        -------
        constrained_vals : numpy.ndarray
            The maximum constrained values of the route
        """
        return self.update_and_evaluate_path(waypoints_flat)[1]

    def unflattened_waypoints(self, waypoinys_flat):
        """
        Unflatten the flattened waypoints

        Parameters
        ----------
        waypoints_flat : numpy.ndarray
            1D array of containting waypoints making up the route that has been passed through numpy.flatten()
        
        Returns
        -------
        waypoints : numpy.ndarray
            2D array of shape (2, N) containting waypoints making up the route
        """
        return waypoinys_flat.reshape(self.n, 2)

    def solve(self, COBYQA_options:dict)->optimize.OptimizeResult:
        """
        Minimizes the route's cost using scipy.minimize.optimize(method='COBYQA').

        Parameters
        ----------
        COBYQA_options : dict
            Dictionary containing the options for the COBYQA function. For more information, see
            https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyqa.html#optimize-minimize-cobyqa 
        
        Returns
        -------
        result : scipy.optimize.OptimizeResult
            Scipy minimizer's result. For more information, see
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult
        """
        initial_waypoints_flat = self.initial_waypoints.flatten()

        bounds_plus_minus = np.concat([
                                np.ones(2)*self.resolution/2, 
                                np.ones(initial_waypoints_flat.size-4)*self.bounds_size/2,
                                np.ones(2)*self.resolution/2
                                ])

        # Create Bounds object
        lower_bound = initial_waypoints_flat - bounds_plus_minus

        upper_bound = initial_waypoints_flat + bounds_plus_minus

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
            method='COBYQA',
            bounds=bounds,
            constraints=nonlinear_constraints,
            options=COBYQA_options
        )
        return result

class RoutePlanner:
    """
    Finds the best route between 2 points on an elevation map

    Parameters
    ----------
    elevation : np.ndarray
        2D array containing the elevation values. 
    x : np.ndarray 
        1D array containing the x coordinates. This array should match with the columns in the inputed elevation array.
    y : np.ndarray
        1D array containing the y coordinates. This array should match with the rows in the inputed elevation array.

    Attributes
    ----------
    elevation : np.ndarray
        2D array containing the elevation values. 
        Elevation is transposed such that row axis (axis=0) is the x axis and the column axis (axis=1) is the y direction.
        This is done to simplify indexing.
    x : np.ndarray 
        1D array containing the x coordinates. x must have constant spacing, and the spacing must be the same with y
    y : np.ndarray
        1D array containing the y coordinates. y must have constant spacing, and the spacing must be the same with x
    resolution : float 
        The spacing between coordinate points.
    """
    def __init__(self, 
                 elevation:np.ndarray, 
                 x:np.ndarray, 
                 y:np.ndarray):

        # Make sure that the shape of the elevation map matches the x and y arrays
        if elevation.shape == (y.size, x.size):
            self.elevation = elevation.T
        else:
            raise Exception("The shape of elevation does not match with x and y.")

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
                      )->tuple: 
        """
        Evaluates the cost and constrained values at a point.      
        
        Parameters
        ----------
        point :  np.ndarray
            2D array of shape (2,) or 2D array of shape (N,2) containing the point to evaluate the cost and constrained values.
        dir : np.ndarray
            2D array of shape (2,N) containing the unit direction vector at each point.  
        x_grad_interp : scipy.interpolate.RegularGridInterpolator,
            A RegularGridInterpolator object that returns the gradient in the x direction
        y_grad_interp : scipy.interpolate.RegularGridInterpolator
            A RegularGridInterpolator object that returns the gradient in the y direction
        relief_interp : scipy.interpolate.RegularGridInterpolator
            A RegularGridInterpolator object that returns the terain relief
        roughness_interp : scipy.interpolate.RegularGridInterpolator
            A RegularGridInterpolator object that returns the terain roughness
        weights : CostWeights
            The weights applied to get the total cost
            
        Returns
        -------
        cost : float
            The cost at the point. The cost is calculated by the following formula:
            cost = (base_cost
            + {uphill_cost if forward gradient is positive, downhill_cost otherwise} * forward_grad
            + side_gradient_cost * abs(sideways_grad) 
            + relief_cost * relief
            + roughness_cost * roughness
            ) 
        
        constrained_vals : numpy.ndarray
            1D array contining [forward_grad, sideways_grad, -forward_grad, relief, roughness]
        """

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
                   grid_factor:int,
                   heuristics_weight,
                   cost_weights:CostWeights, 
                   constraints:np.ndarray,
                   line_of_sight_optimization:bool=True,
                   minimizer_optimization:bool=False,
                   minimizer_bounds_size:float=1000,
                   minimizer_max_iter:int=500,
                   minimizer_constraint_tol:float= 1e-4, 
                   minimizer_waypoints_tol:float= 1
                )->tuple:
        """
        Finds the best route between 2 points through the elevation map    
        
        Parameters
        ----------
        start : np.ndarray 
            1D array of shape (2,) containing the starting coordinates
        goal : np.ndarray 
            1D array of shape (2,) containing the goal coordinates
        grid_factor : int
            The factor to sample the grid for A* search. A grid_factor of 3 means only 1 every 3 coordinate points in the x and y direction is used, meaning only 1 out of 9 points on the grid are used.
        heuristics_weight : float
            The weight for the heuristic term h(x) of the A* search
        cost_weights : CostWeights 
            The weights for cost calculation
        constraints : np.ndarray
            The constraints vector. Can be made using make_constraints_vector()
        line_of_sight_optimization : bool, optional
            Set to True to apply line of sight optimization (highly recomended). Defaults to True.
        minimizer_optimization : bool, optional
            Set to True to apply minimizer optimization. Very computationally expensive. 
            Using minimizer optimization without line of sight optimization is not recomended. Defaults to False. 
        minimizer_bounds_size : float, optional
            The size of the boundsfor the minimizer. The minimizer will only evaluate points inside of an (n unit x n unit) window arround the initial waypoints.
            Defaults to 1000.
        minimizer_max_iter : int, optional
            Maximum number of iterations for the minimizer algorithm. Defaults to 500
        minimizer_constraint_tol : float, optional 
            Tolarance for the constraints value for the minimizer. Defaults to 1e-4
        minimizer_waypoints_tol : float, optional
            Tolerance for the final values for waypoints for the minimizer. Defaults to 1. 
            
        Returns
        -------
        waypoints : numpy.ndarray
            ndarray of shape (N,2). Contains the waypoints making up the route if A* search is succesfull. 
            Contains all of the visited cells if A* search is unsuccessfull.  
        cost : float
            The final cost of the route. None if search is unsuccessful.
        a_star_success_flag : bool
            True if A* search is successfull.
        minimizer_success_flag : bool
            True if the minimizer successfully converges
        """

        x_grad, y_grad = np.gradient(self.elevation, self.x, self.y)
        x_grad_interp = RegularGridInterpolator((self.x,self.y), x_grad, bounds_error=False, fill_value=np.nan) 
        y_grad_interp = RegularGridInterpolator((self.x,self.y), y_grad, bounds_error=False, fill_value=np.nan) 

        relief = ndimage.maximum_filter(self.elevation, cost_weights.window_size) - ndimage.minimum_filter(self.elevation, cost_weights.window_size)
        relief_interp = RegularGridInterpolator((self.x,self.y), relief, bounds_error=False, fill_value=np.nan)

        roughness = np.sqrt(
            ndimage.uniform_filter(
                np.square(
                    self.elevation - ndimage.uniform_filter(self.elevation, size=cost_weights.window_size)
                )
            , size=cost_weights.window_size)
        )
        roughness_interp = RegularGridInterpolator((self.x,self.y), roughness, bounds_error=False, fill_value=np.nan)

        print("Using A* search")

        eval_func = lambda point, dir: self.eval_function(
            point=point,
            dir=dir,
            x_grad_interp=x_grad_interp,
            y_grad_interp=y_grad_interp,
            relief_interp=relief_interp,
            roughness_interp=roughness_interp,
            weights=cost_weights
        )

        a_star_route, a_star_cost, a_star_success = a_star_search(
            start=start,
            goal=goal,
            x=self.x[int(grid_factor/2)::grid_factor],
            y=self.y[int(grid_factor/2)::grid_factor],
            eval_func=eval_func,
            eval_resolution=self.resolution,
            constraints=constraints,
            heuristic_weight=heuristics_weight,
            apply_line_of_sight_optimization=line_of_sight_optimization
            )

        if not a_star_success:
            print("A* search failed to find goal")
            return a_star_route, None, False, False

        if not minimizer_optimization:
            return a_star_route, a_star_cost, True, False

        print("Applying minimizer optimization")
        
        route_optimizer = MinimizeRouteCost(
            initial_waypoints=a_star_route,
            eval_func=eval_func,
            constraints=constraints,
            resolution=self.resolution,
            bounds_size=minimizer_bounds_size
        )

        optimizer_result = route_optimizer.solve({
            'disp':False,
            'maxiter':minimizer_max_iter,
            'feasibility_tol':minimizer_constraint_tol,
            'final_tr_radius':minimizer_waypoints_tol,
            'initial_tr_radius':minimizer_waypoints_tol
        })

        if not optimizer_result.success:
            print(f"Minimizer failed to converge, reason: {optimizer_result.message}")

        print(f"Optimization completed with cost of {optimizer_result.fun:.4g}")

        return (
            route_optimizer.unflattened_waypoints(optimizer_result.x),
            optimizer_result.fun,
            True,
            True
        )