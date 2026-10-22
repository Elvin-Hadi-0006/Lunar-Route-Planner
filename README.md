# Lunar Route Planner
This repository is part of the final project for MMA3001 - Numerical methods and machine learning at Monash University. This project is meant to provide a way to find the best route between 2 points on the moon using SLDEM lunar elevation data.

# How it works
The RoutePlanner.find_route() method applies the A* search algorithm on the elevation map. Cost and barriers are determined by gradient, relief, and roughness. The a line-of-sight is then applied to the A* result to remove redundant waypoints. A multivariate minimizer optimization ( [scipy.optimize.minimize(method=COBYQA)](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyqa.html#optimize-minimize-cobyqa)) can also be applied to the result, though this greatly increases computation time.

# Required External Libraries
For src/
* [numpy](https://numpy.org/) 
* [scipy](https://scipy.org/)
* [rasterio](https://rasterio.readthedocs.io/en/stable/)
* [pvl](https://pvl.readthedocs.io/en/stable/readme.html)

For test/
* [matplotlib](https://matplotlib.org/) 
* [pillow](https://pypi.org/project/pillow/)

# Dataset
Thttps://pillow.readthedocs.io/en/stable/h of lunar digital elevation models. The model is a combination of data from the Lunar Orbiter Laser Altimeter (LOLA) data and Selenological and Engineering Explorer (SELENE or Kaguya, operated by the Japan Aerospace eXploration Agency). For more information on the dataseet, visit [here](https://ode.rsl.wustl.edu/moon/pagehelp/Content/Missions_Instruments/Lunar%20Reconnaissance%20Orbiter%20(LRO)/LOLA/GDR/SLDEM.htm). 

The test functtions uses the sldem2015_512_00n_30n_315_360_float dataset. This dataset and and other SLDEM datasets are available at the [Lunar Orbital Data Explorer](https://ode.rsl.wustl.edu/moon/download) website.

# AI Statement
Parts of this repository were developed with the assistance of AI tools such as Claude Sonnet 5.5.

All content has been reviewed, verified, and adapted by the author to ensure accuracy, clarity, and alignment with the learning objectives.
