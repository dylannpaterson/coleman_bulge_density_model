import numpy as np
from astropy.io import fits
import json
import os

def compile_data_for_shipping(fits_file, json_file, output_file):
    print("Extracting FITS data...")
    with fits.open(fits_file) as hdul:
        header = hdul[0].header
        data = hdul[0].data.astype(float)

    # Calculate axes
    r_axis = header['CRVAL3'] + (np.arange(header['NAXIS3']) - (header['CRPIX3'] - 1)) * header['CDELT3']
    lat_axis = header['CRVAL2'] + (np.arange(header['NAXIS2']) - (header['CRPIX2'] - 1)) * header['CDELT2']
    lon_axis = header['CRVAL1'] + (np.arange(header['NAXIS1']) - (header['CRPIX1'] - 1)) * header['CDELT1']

    if lon_axis[0] > lon_axis[-1]:
        lon_axis = lon_axis[::-1]
        data = np.flip(data, axis=2)

    # Load SX params
    with open(json_file, 'r') as f:
        sx_params = json.load(f)["base_case"]

    print(f"Compressing and saving to {output_file}...")
    np.savez_compressed(
        output_file, 
        density_grid=data, 
        r_axis=r_axis, 
        lat_axis=lat_axis, 
        lon_axis=lon_axis,
        **sx_params 
    )
    print("Done! The .npz file is ready for packaging.")

if __name__ == '__main__':
    # Save the output directly into the package directory
    output_path = os.path.join('coleman_bulge_density', 'model_data.npz')
    
    compile_data_for_shipping(
        fits_file='data/vvvdaophot_density_symmetric_mlferr_spiralarmbg_20x20.fits',
        json_file='sx_parameters.json',
        output_file=output_path
    )