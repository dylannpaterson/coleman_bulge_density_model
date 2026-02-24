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

    # We use @jax.jit to compile this entire function for massive speedups

    def __call__(self, r=None, lat=None, lon=None, x=None, y=None, z=None):
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
        
        dens_out = density.reshape(original_shape)
        bound_out = in_bounds.reshape(original_shape)
        
        return dens_out, bound_out

    def _sx_model_extrapolate(self, r, lat, lon):
        p = self.sx_params 
        
        lon_rad = jnp.deg2rad(lon + 180.0) 
        lat_rad = jnp.deg2rad(lat)
        
        x_sun = r * jnp.cos(lat_rad) * jnp.cos(lon_rad)
        y_sun = r * jnp.cos(lat_rad) * jnp.sin(lon_rad)
        z_sun = r * jnp.sin(lat_rad)

        x_gc = x_sun + 8.0 + p['delta_R_0']
        y_gc = y_sun
        z_gc = z_sun + p['z_sun_offset']

        alpha_rad = jnp.deg2rad(p['alpha'])
        X = x_gc * jnp.cos(alpha_rad) + y_gc * jnp.sin(alpha_rad)
        Y = -x_gc * jnp.sin(alpha_rad) + y_gc * jnp.cos(alpha_rad)
        Z = z_gc

        r1_c_parallel = (((jnp.abs(X) / p['x0'])**p['c_perp'] + (jnp.abs(Y) / p['y0'])**p['c_perp'])**(p['c_parallel'] / p['c_perp']) + (jnp.abs(Z) / p['z0'])**p['c_parallel'])
        r1 = r1_c_parallel**(1 / p['c_parallel'])

        r2 = jnp.sqrt(((jnp.abs(X - p['C'] * Z)) / p['x1'])**2 + (jnp.abs(Y) / p['y1'])**2)
        r3 = jnp.sqrt(((jnp.abs(X + p['C'] * Z)) / p['x1'])**2 + (jnp.abs(Y) / p['y1'])**2)
        
        density = p['rho0'] / jnp.cosh(r1)**2 * (1 + p['A'] * (jnp.exp(-r2**p['n']) + jnp.exp(-r3**p['n'])))
        
        R = jnp.sqrt(X**2 + Y**2)
        R_cutoff = 4.5 
        exp_cutoff = jnp.where(R < R_cutoff, 1.0, jnp.exp(-2 * (R - R_cutoff)**2))
        
        return density * exp_cutoff

    @staticmethod
    def xyz_to_r_lat_lon(x, y, z):
        x_sun, y_sun, z_sun = jnp.asarray(x) - 8.0, jnp.asarray(y), jnp.asarray(z)
        r = jnp.sqrt(x_sun**2 + y_sun**2 + z_sun**2)
        
        safe_r = jnp.where(r == 0, 1.0, r)
        
        lat = jnp.rad2deg(jnp.arcsin(z_sun / safe_r))
        lon = jnp.rad2deg(jnp.arctan2(y_sun, x_sun)) - 180.0
        lon = jnp.where(lon < -180, lon + 360, lon)
        
        lat = jnp.where(r == 0, 0.0, lat)
        lon = jnp.where(r == 0, 0.0, lon)
        
        return r, lat, lon

# --- INSTANTIATE ON IMPORT ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
_data_path = os.path.join(_current_dir, 'model_data.npz')
bulge_density_model = ColemanBulgeDensityModel(_data_path)