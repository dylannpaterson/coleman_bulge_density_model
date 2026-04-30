import os
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
import functools

# It is highly recommended to enable 64-bit precision for astrophysics models 
# to prevent numerical instability during coordinate transformations and gradient calculation.
jax.config.update("jax_enable_x64", True)

class ColemanBulgeDensityModel:
    def __init__(self, npz_path):
        """
        Initializes the density model by loading pre-computed data from a .npz file.
        """
        # Load the data using standard numpy first
        data = np.load(npz_path)
        
        # Convert grid arrays to JAX arrays
        self.density_grid = jnp.array(data['density_grid'])
        
        self.r_axis = jnp.array(data['r_axis'])
        self.lat_axis = jnp.array(data['lat_axis'])
        self.lon_axis = jnp.array(data['lon_axis'])
        
        # Calculate grid boundaries and steps for map_coordinates index mapping
        self.r_start, self.r_end = self.r_axis[0], self.r_axis[-1]
        self.r_step = self.r_axis[1] - self.r_axis[0]
        
        self.lat_start, self.lat_end = self.lat_axis[0], self.lat_axis[-1]
        self.lat_step = self.lat_axis[1] - self.lat_axis[0]
        
        self.lon_start, self.lon_end = self.lon_axis[0], self.lon_axis[-1]
        self.lon_step = self.lon_axis[1] - self.lon_axis[0]

        # Extract SX parameters, safely handling non-numeric metadata strings
        exclude_keys = {'density_grid', 'r_axis', 'lat_axis', 'lon_axis'}
        self.sx_params = {}
        for k in data.files:
            if k not in exclude_keys:
                val = data[k].item()
                try:
                    # Convert math parameters to JAX floats
                    self.sx_params[k] = jnp.float64(val)
                except ValueError:
                    # Leave strings (like "(A) Base case") as they are
                    self.sx_params[k] = val
        
        # Override with optimized symmetry parameters for the symmetric version
        self.sym_params = {
            'dx': 7.899703,
            'dy': -0.020903,
            'dz': 0.0,
            'alpha': -18.430253
        }

    # We use @jax.jit to compile this entire function for massive speedups

    def __call__(self, r=None, lat=None, lon=None, x=None, y=None, z=None):
        """
        Evaluates the Milky Way bulge density at given coordinates.

        The returned densities are in units of stars per kpc^3, normalised to a 
        total bulge stellar count of 30.7 x 10^9, as derived from the bulge 
        luminosity function.

        Users can provide coordinates in either Sun-centered spherical 
        (r, lat, lon) or Galactic Cartesian (x, y, z) systems.

        Args:
            r (array_like, optional): Radius from the Sun (kpc).
            lat (array_like, optional): Galactic latitude (degrees).
            lon (array_like, optional): Galactic longitude (degrees).
            x (array_like, optional): Cartesian X coordinate (kpc).
            y (array_like, optional): Cartesian Y coordinate (kpc).
            z (array_like, optional): Cartesian Z coordinate (kpc).

        Returns:
            A tuple containing:
                - density (jax.numpy.ndarray): The bulge density at the given coordinates.
                - in_bounds (jax.numpy.ndarray): A boolean array indicating whether 
                  the coordinates were within the pre-computed grid.
        """
        is_spherical = (r is not None) and (lat is not None) and (lon is not None)
        is_cartesian = (x is not None) and (y is not None) and (z is not None)

        # Handle shapes and coordinate conversions purely in Python
        if is_spherical and not is_cartesian:
            original_shape = jnp.shape(r)
        elif is_cartesian and not is_spherical:
            original_shape = jnp.shape(x)
            r, lat, lon = self.xyz_to_r_lat_lon(x, y, z)
        else:
            raise ValueError(
                "Provide either (r, lat, lon) for spherical coordinates or (x, y, z) "
                "for Cartesian coordinates, but not a mix of both."
            )

        # Flatten input arrays
        r_flat = jnp.atleast_1d(r).ravel()
        lat_flat = jnp.atleast_1d(lat).ravel()
        lon_flat = jnp.atleast_1d(lon).ravel()
        
        # Pass the prepared arrays to the ultra-fast JAX-compiled core
        return self._evaluate_core(r_flat, lat_flat, lon_flat, original_shape)

    def evaluate_symmetric(self, r=None, lat=None, lon=None, x=None, y=None, z=None):
        """
        Evaluates the symmetric version of the bulge density by averaging over 
        the 8 octants in the optimized symmetry frame.
        """
        is_spherical = (r is not None) and (lat is not None) and (lon is not None)
        is_cartesian = (x is not None) and (y is not None) and (z is not None)

        if is_spherical and not is_cartesian:
            original_shape = jnp.shape(r)
            x_sun, y_sun, z_sun = self.r_lat_lon_to_xyz_sun(r, lat, lon)
        elif is_cartesian and not is_spherical:
            original_shape = jnp.shape(x)
            x_sun, y_sun, z_sun = jnp.asarray(x) - 8.0, jnp.asarray(y), jnp.asarray(z)
        else:
            raise ValueError("Provide either (r, lat, lon) or (x, y, z).")

        x_flat = jnp.atleast_1d(x_sun).ravel()
        y_flat = jnp.atleast_1d(y_sun).ravel()
        z_flat = jnp.atleast_1d(z_sun).ravel()

        density = self._evaluate_symmetric_core(x_flat, y_flat, z_flat, original_shape)
        return density

    @functools.partial(jax.jit, static_argnames=['self', 'original_shape'])
    def _evaluate_symmetric_core(self, x_sun, y_sun, z_sun, original_shape):
        p = self.sym_params
        alpha_rad = jnp.deg2rad(p['alpha'])

        # 1. Transform to symmetry frame (X_s, Y_s, Z_s)
        x_gc = x_sun + p['dx']
        y_gc = y_sun + p['dy']
        z_gc = z_sun + p['dz']

        Xs = x_gc * jnp.cos(alpha_rad) + y_gc * jnp.sin(alpha_rad)
        Ys = -x_gc * jnp.sin(alpha_rad) + jnp.cos(alpha_rad) * y_gc
        Zs = z_gc

        # 2. Evaluate Grid Part (8-way average)
        signs = jnp.array([
            [1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1],
            [-1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]
        ], dtype=jnp.float64)

        def get_octant_data(s):
            # Apply symmetry in SX frame
            X_s = Xs * s[0]
            Y_s = Ys * s[1]
            Z_s = Zs * s[2]

            # Rotate back to GC
            x_gc_s = X_s * jnp.cos(alpha_rad) - Y_s * jnp.sin(alpha_rad)
            y_gc_s = X_s * jnp.sin(alpha_rad) + Y_s * jnp.cos(alpha_rad)
            z_gc_s = Z_s

            # Back to Sun-centered
            x_s = x_gc_s - p['dx']
            y_s = y_gc_s - p['dy']
            z_s = z_gc_s - p['dz']

            r, lat, lon = self._xyz_sun_to_r_lat_lon(x_s, y_s, z_s)
            
            dens = self._evaluate_grid_only(r, lat, lon)
            in_b = (r >= self.r_start) & (r <= self.r_end) & \
                   (lat >= self.lat_start) & (lat <= self.lat_end) & \
                   (lon >= self.lon_start) & (lon <= self.lon_end)
            return dens, in_b

        grid_densities, in_bounds_all = jax.vmap(get_octant_data)(signs)
        # Avoid division by zero: if none are in bounds, this won't be used anyway due to in_bounds_any
        sum_grid_density = jnp.sum(grid_densities, axis=0)
        sum_in_bounds = jnp.sum(in_bounds_all.astype(jnp.float64), axis=0)
        mean_grid_density = sum_grid_density / jnp.where(sum_in_bounds > 0, sum_in_bounds, 1.0)
        in_bounds_any = jnp.any(in_bounds_all, axis=0)

        # 3. Evaluate Analytical Extrapolation (already symmetric, evaluate once)
        extrap_density = self._sx_model_extrapolate_sx(jnp.abs(Xs), jnp.abs(Ys), jnp.abs(Zs))

        # 4. Combine
        norm_factor = 31735.587
        density = jnp.where(in_bounds_any, mean_grid_density, extrap_density)
        
        return (density * norm_factor).reshape(original_shape)

    @functools.partial(jax.jit, static_argnames=['self'])
    def _evaluate_grid_only(self, r, lat, lon):
        r_idx = (r - self.r_start) / self.r_step
        lat_idx = (lat - self.lat_start) / self.lat_step
        lon_idx = (lon - self.lon_start) / self.lon_step
        coords = jnp.stack([r_idx, lat_idx, lon_idx])
        
        # We need to handle out-of-bounds carefully to return 0 for the mean
        return map_coordinates(self.density_grid, coords, order=1, mode='constant', cval=0.0)

    def _sx_model_extrapolate_sx(self, Xs, Ys, Zs):
        """Symmetric version of the SX model extrapolation."""
        p = self.sx_params
        # Use absolute values to handle negative coordinates (important for non-symmetric mode)
        X, Y, Z = jnp.abs(Xs), jnp.abs(Ys), jnp.abs(Zs)

        r1_c_parallel = (((X / p['x0'])**p['c_perp'] + (Y / p['y0'])**p['c_perp'])**(p['c_parallel'] / p['c_perp']) + (Z / p['z0'])**p['c_parallel'])
        r1 = r1_c_parallel**(1 / p['c_parallel'])

        r2 = jnp.sqrt(((jnp.abs(X - p['C'] * Z)) / p['x1'])**2 + (Y / p['y1'])**2)
        r3 = jnp.sqrt(((jnp.abs(X + p['C'] * Z)) / p['x1'])**2 + (Y / p['y1'])**2)
        
        return p['rho0'] / jnp.cosh(r1)**2 * (1 + p['A'] * (jnp.exp(-r2**p['n']) + jnp.exp(-r3**p['n'])))

    # 3. Add this new method. We tell JAX that 'self' and 'original_shape' 
    # are static (do not try to turn them into arrays).
    @functools.partial(jax.jit, static_argnames=['self', 'original_shape'])
    def _evaluate_core(self, r_flat, lat_flat, lon_flat, original_shape):
        
        # Identify bounds
        in_bounds = (
            (r_flat >= self.r_start) & (r_flat <= self.r_end) &
            (lat_flat >= self.lat_start) & (lat_flat <= self.lat_end) &
            (lon_flat >= self.lon_start) & (lon_flat <= self.lon_end)
        )
        
        # --- 1. Evaluate Grid Interpolation ---
        r_idx = (r_flat - self.r_start) / self.r_step
        lat_idx = (lat_flat - self.lat_start) / self.lat_step
        lon_idx = (lon_flat - self.lon_start) / self.lon_step
        
        coords = jnp.stack([r_idx, lat_idx, lon_idx])
        
        interp_density = map_coordinates(
            self.density_grid, coords, order=1, mode='nearest'
        )
            
        # --- 2. Evaluate Analytical Extrapolation ---
        extrap_density = self._sx_model_extrapolate(r_flat, lat_flat, lon_flat)
            
        # --- 3. Combine without branching ---
        density = jnp.where(in_bounds, interp_density, extrap_density)

        # Normalise the density model to a total bulge stellar count of 30.7 x 10^9.
        # This normalization was derived by integrating the raw density 
        # (unit: [RC stars / pc^3 / sr]) over all space and scaling to the target.
        # Factor = (30.7 x 10^9) / Integral(raw_density) = 31735.587
        norm_factor = 31735.587
        density = density * norm_factor
        
        dens_out = density.reshape(original_shape)
        bound_out = in_bounds.reshape(original_shape)
        
        return dens_out, bound_out

    def _sx_model_extrapolate(self, r, lat, lon):
        p = self.sx_params 
        
        r_val, lat_val, lon_val = r, lat, lon
        x_sun, y_sun, z_sun = self.r_lat_lon_to_xyz_sun(r_val, lat_val, lon_val)

        # Use optimized offsets if available, otherwise fallback
        x_gc = x_sun + p.get('dx_sun', 8.0 + p.get('delta_R_0', 0.0))
        y_gc = y_sun + p.get('dy_sun', 0.0)
        z_gc = z_sun + p.get('dz_sun', p.get('z_sun_offset', 0.0))

        alpha_rad = jnp.deg2rad(p['alpha'])
        X = x_gc * jnp.cos(alpha_rad) + y_gc * jnp.sin(alpha_rad)
        Y = -x_gc * jnp.sin(alpha_rad) + y_gc * jnp.cos(alpha_rad)
        Z = z_gc

        return self._sx_model_extrapolate_sx(X, Y, Z)

    @staticmethod
    def xyz_to_r_lat_lon(x, y, z):
        x_sun, y_sun, z_sun = jnp.asarray(x) - 8.0, jnp.asarray(y), jnp.asarray(z)
        return ColemanBulgeDensityModel._xyz_sun_to_r_lat_lon(x_sun, y_sun, z_sun)

    @staticmethod
    def _xyz_sun_to_r_lat_lon(x_sun, y_sun, z_sun):
        r = jnp.sqrt(x_sun**2 + y_sun**2 + z_sun**2)
        safe_r = jnp.where(r == 0, 1.0, r)
        lat = jnp.rad2deg(jnp.arcsin(z_sun / safe_r))
        lon = jnp.rad2deg(jnp.arctan2(y_sun, x_sun)) - 180.0
        lon = jnp.where(lon < -180, lon + 360, lon)
        lat = jnp.where(r == 0, 0.0, lat)
        lon = jnp.where(r == 0, 0.0, lon)
        return r, lat, lon

    @staticmethod
    def r_lat_lon_to_xyz_sun(r, lat, lon):
        lon_rad = jnp.deg2rad(lon + 180.0) 
        lat_rad = jnp.deg2rad(lat)
        x_sun = r * jnp.cos(lat_rad) * jnp.cos(lon_rad)
        y_sun = r * jnp.cos(lat_rad) * jnp.sin(lon_rad)
        z_sun = r * jnp.sin(lat_rad)
        return x_sun, y_sun, z_sun

# --- INSTANTIATE ON IMPORT ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
_data_path = os.path.join(_current_dir, 'model_data.npz')
bulge_density_model = ColemanBulgeDensityModel(_data_path)