import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator

class ColemanBulgeDensityModel:
    def __init__(self, npz_path):
        """
        Initializes the density model by loading pre-computed data from a .npz file.

        This approach allows for rapid instantiation, as the expensive computations
        for the density grid have already been performed and saved.

        Args:
            npz_path (str): The file path to the NumPy .npz data archive.
        """
        # Load the compressed NumPy data file.
        data = np.load(npz_path)
        
        # Extract the coordinate axes for the grid. These define the space
        # over which the density grid is defined.
        self.r_axis = data['r_axis']       # Radial distance axis
        self.lat_axis = data['lat_axis']   # Latitude axis
        self.lon_axis = data['lon_axis']   # Longitude axis
        self.axes = (self.r_axis, self.lat_axis, self.lon_axis)
        
        # Reconstruct the sx_params dictionary from the npz file.
        # These are the parameters for the analytical (extrapolation) model.
        # The .item() method is used to convert 0-dimensional NumPy arrays
        # back into standard Python scalar types (e.g., floats).
        exclude_keys = {'density_grid', 'r_axis', 'lat_axis', 'lon_axis'}
        self.sx_params = {k: data[k].item() for k in data.files if k not in exclude_keys}

        # Set up the SciPy RegularGridInterpolator. This is the core engine
        # for quickly retrieving density values from within the pre-computed grid.
        # - `bounds_error=False`: Prevents raising an error for points outside the grid.
        # - `fill_value=None`: For out-of-bounds points, this tells the interpolator
        #   to not use a fixed value, allowing us to handle them separately.
        self.interpolator = RegularGridInterpolator(
            self.axes, data['density_grid'], bounds_error=False, fill_value=None
        )

    def __call__(self, r=None, lat=None, lon=None, x=None, y=None, z=None):
        """
        Evaluates the density at given coordinates, accepting either Sun-centered
        spherical (r, lat, lon) or Galactic Cartesian (x, y, z) inputs.

        This method is vectorized, meaning it can efficiently process large arrays
        of coordinates at once. It intelligently switches between interpolation for
        points inside its pre-computed grid and extrapolation for points outside.

        Args:
            r (float or np.ndarray, optional): Radial distance(s) from the origin.
            lat (float or np.ndarray, optional): Latitude(s).
            lon (float or np.ndarray, optional): Longitude(s).
            x (float or np.ndarray, optional): Galactic Cartesian x-coordinate(s).
            y (float or np.ndarray, optional): Galactic Cartesian y-coordinate(s).
            z (float or np.ndarray, optional): Galactic Cartesian z-coordinate(s).

        Returns:
            tuple: A pair of (density, in_bounds) values.
                - density (float or np.ndarray): The calculated density at each point.
                - in_bounds (bool or np.ndarray): A boolean indicating if each point
                  was within the interpolation grid.
        
        Raises:
            ValueError: If the provided coordinate system is ambiguous or incomplete.
        """
        # --- Input Coordinate Handling ---
        is_spherical = (r is not None) and (lat is not None) and (lon is not None)
        is_cartesian = (x is not None) and (y is not None) and (z is not None)

        if is_spherical and not is_cartesian:
            # Proceed with the provided spherical coordinates.
            pass
        elif is_cartesian and not is_spherical:
            # Convert Cartesian to spherical before proceeding.
            r, lat, lon = self.xyz_to_r_lat_lon(x, y, z)
        else:
            raise ValueError(
                "Provide either (r, lat, lon) for spherical coordinates or (x, y, z) "
                "for Cartesian coordinates, but not a mix of both."
            )

        # Ensure inputs are at least 1D NumPy arrays for consistent processing.
        r = np.atleast_1d(r).astype(float)
        lat = np.atleast_1d(lat).astype(float)
        lon = np.atleast_1d(lon).astype(float)
        
        # Store original shape to reshape the output correctly.
        original_shape = r.shape
        # Flatten input arrays for efficient, vectorized calculations.
        r_flat, lat_flat, lon_flat = r.ravel(), lat.ravel(), lon.ravel()
        
        # Initialize an array to store density results.
        density = np.zeros_like(r_flat)
        
        # Perform a vectorized check to determine which points are inside the
        # pre-defined interpolation grid bounds.
        in_bounds = (
            (r_flat >= self.r_axis.min()) & (r_flat <= self.r_axis.max()) &
            (lat_flat >= self.lat_axis.min()) & (lat_flat <= self.lat_axis.max()) &
            (lon_flat >= self.lon_axis.min()) & (lon_flat <= self.lon_axis.max())
        )
        
        # --- Branch 1: Process points INSIDE the interpolation grid ---
        if np.any(in_bounds):
            # Group the coordinates of in-bounds points into a single array.
            points = np.column_stack((r_flat[in_bounds], lat_flat[in_bounds], lon_flat[in_bounds]))
            # Use the pre-initialized interpolator to calculate density for these points.
            density[in_bounds] = self.interpolator(points)
            
        # --- Branch 2: Process points OUTSIDE the grid using an analytical model ---
        out_bounds = ~in_bounds
        if np.any(out_bounds):
            # For points outside the grid, fall back to the analytical SX model.
            density[out_bounds] = self._sx_model_extrapolate(
                r_flat[out_bounds], lat_flat[out_bounds], lon_flat[out_bounds]
            )
            
        # Reshape the flat result arrays back to the original input shape.
        dens_out = density.reshape(original_shape)
        bound_out = in_bounds.reshape(original_shape)
        
        # If the original input was a scalar, return scalar values.
        if dens_out.ndim == 0:
            return dens_out.item(), bound_out.item()
        return dens_out, bound_out

    def _sx_model_extrapolate(self, r, lat, lon):
        """
        Calculates density using the analytical "SX" model.

        This method serves as a fallback for coordinates that lie outside the
        pre-computed interpolation grid. It's a fully vectorized implementation
        of the Coleman et al. (2020) bulge/bar model.

        Args:
            r (np.ndarray): Radial distance(s).
            lat (np.ndarray): Latitude(s).
            lon (np.ndarray): Longitude(s).

        Returns:
            np.ndarray: The calculated density at each point.
        """
        p = self.sx_params # Unpack the model parameters for easier access.
        
        # --- Coordinate Transformations ---
        # Convert spherical (lat, lon) to radians for trigonometric functions.
        lon_rad = np.deg2rad(lon + 180.0) # Shift longitude for the model's frame.
        lat_rad = np.deg2rad(lat)
        
        # Convert from Sun-centered spherical coordinates to Sun-centered Cartesian.
        x_sun = r * np.cos(lat_rad) * np.cos(lon_rad)
        y_sun = r * np.cos(lat_rad) * np.sin(lon_rad)
        z_sun = r * np.sin(lat_rad)

        # Shift to Galactic-centered Cartesian coordinates.
        # This accounts for the Sun's position relative to the Galactic center.
        x_gc = x_sun + 8.0 + p['delta_R_0']
        y_gc = y_sun
        z_gc = z_sun + p['z_sun_offset']

        # Rotate the coordinate system around the Z-axis to align with the Galactic bar.
        alpha_rad = np.deg2rad(p['alpha'])
        X = x_gc * np.cos(alpha_rad) + y_gc * np.sin(alpha_rad)
        Y = -x_gc * np.sin(alpha_rad) + y_gc * np.cos(alpha_rad)
        Z = z_gc

        # --- Density Calculation using the SX Model Formulae ---
        # These equations define the geometry and density distribution of the model.
        
        # r1 defines the main "bar" or "box" component of the bulge.
        r1_c_parallel = (((np.abs(X) / p['x0'])**p['c_perp'] + (np.abs(Y) / p['y0'])**p['c_perp'])**(p['c_parallel'] / p['c_perp']) + (np.abs(Z) / p['z0'])**p['c_parallel'])
        r1 = r1_c_parallel**(1 / p['c_parallel'])

        # r2 and r3 define the "X-shape" or "peanut" structure.
        r2 = np.sqrt(((np.abs(X - p['C'] * Z)) / p['x1'])**2 + (np.abs(Y) / p['y1'])**2)
        r3 = np.sqrt(((np.abs(X + p['C'] * Z)) / p['x1'])**2 + (np.abs(Y) / p['y1'])**2)
        
        # Combine the components into the final density equation.
        # It consists of a sech^2 (hyperbolic secant) profile for the bar
        # and an enhancement term for the X-shape.
        density = p['rho0'] / np.cosh(r1)**2 * (1 + p['A'] * (np.exp(-r2**p['n']) + np.exp(-r3**p['n'])))
        
        # Apply a smooth cutoff at large radii to prevent unrealistic density values far from the center.
        R = np.sqrt(X**2 + Y**2)
        R_cutoff = 4.5 # The radius at which the cutoff begins.
        # `np.where` is used to apply the cutoff only where R exceeds the threshold.
        exp_cutoff = np.where(R < R_cutoff, 1.0, np.exp(-2 * (R - R_cutoff)**2))
        
        return density * exp_cutoff

    @staticmethod
    def xyz_to_r_lat_lon(x, y, z):
        """
        Converts Galactic Cartesian coordinates (x, y, z) to Sun-centered
        spherical coordinates (r, lat, lon).

        This is a utility function to transform from a standard Cartesian frame
        to the spherical frame used by this density model. It is fully vectorized
        to handle arrays of coordinates efficiently.

        Args:
            x (float or np.ndarray): Cartesian x-coordinate(s).
            y (float or np.ndarray): Cartesian y-coordinate(s).
            z (float or np.ndarray): Cartesian z-coordinate(s).

        Returns:
            tuple: A tuple containing (r, lat, lon) arrays.
        """
        # Convert inputs to NumPy arrays and shift origin from Galactic Center to the Sun's position.
        # The Sun is assumed to be at x=8.0, y=0, z=0 in this Galactic frame.
        x_sun, y_sun, z_sun = np.asarray(x) - 8.0, np.asarray(y), np.asarray(z)
        
        # Calculate radial distance 'r' from the Sun.
        r = np.sqrt(x_sun**2 + y_sun**2 + z_sun**2)
        
        # --- Handle the singularity at the origin (r=0) ---
        # To avoid division by zero in lat/lon calculations, temporarily set r=1 where r=0.
        safe_r = np.where(r == 0, 1.0, r)
        
        # Calculate latitude (angle from the xy-plane).
        lat = np.rad2deg(np.arcsin(z_sun / safe_r))
        
        # Calculate longitude (angle in the xy-plane from the x-axis).
        # The -180 degree shift aligns the coordinate system with model expectations.
        lon = np.rad2deg(np.arctan2(y_sun, x_sun)) - 180.0
        # Normalize longitude to the range [-180, 180).
        lon = np.where(lon < -180, lon + 360, lon)
        
        # At the exact origin (r=0), latitude and longitude are undefined.
        # We assign them a conventional value of 0.
        lat = np.where(r == 0, 0.0, lat)
        lon = np.where(r == 0, 0.0, lon)
        return r, lat, lon

# --- INSTANTIATE ON IMPORT ---
# The following code runs once when this module is first imported.

# Get the absolute path of the directory containing this script.
_current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to the required data file, assuming it's in the same directory.
_data_path = os.path.join(_current_dir, 'model_data.npz')

# Create a single, ready-to-use instance of the density model.
bulge_density_model = ColemanBulgeDensityModel(_data_path)