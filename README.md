# Coleman Bulge Density Model

A vectorized implementation of the Coleman et al. (2020) Milky Way bulge density model.

## Features

* **Hybrid Evaluation Engine**: Rapidly evaluates density using SciPy's `RegularGridInterpolator` for coordinates within a pre-computed grid. It gracefully falls back to an analytical "SX" model for extrapolation outside the grid bounds.
* **Vectorized Processing**: Fully vectorized to efficiently process large NumPy arrays of coordinates at once.
* **Flexible Coordinate Systems**: Accepts inputs in either Sun-centered spherical coordinates `(r, lat, lon)` or Galactic Cartesian coordinates `(x, y, z)`. 
* **Pre-instantiated**: The package automatically instantiates the model and loads the pre-computed grid data (`model_data.npz`) upon import, making it ready to use immediately.

## Requirements

* Python >= 3.8
* NumPy >= 1.20.0
* SciPy >= 1.7.0

## Installation

You can install this package locally using `pip`:

```bash
pip install coleman_bulge_density_model

```

## Examples

``` python

import numpy as np
from coleman_bulge_density import bulge_density_model

# 1. Using Sun-centered spherical coordinates (r, lat, lon)
density, in_bounds = bulge_density_model(r=8.0, lat=0.0, lon=0.0)

# 2. Using Galactic Cartesian coordinates (x, y, z)
# Note: Do not mix coordinate systems.
density, in_bounds = bulge_density_model(x=8.0, y=0.0, z=0.0)

# 3. Processing large arrays
r_arr = np.linspace(0, 15, 100)
lat_arr = np.zeros(100)
lon_arr = np.zeros(100)
densities, bounds = bulge_density_model(r=r_arr, lat=lat_arr, lon=lon_arr)

```