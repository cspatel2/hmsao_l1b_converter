#%%
#this file creates the line profile of a chosen emission line from a given l1a file, to use as input to the secondary straightening process
import xarray as xr
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import matplotlib.pyplot as plt
#%%

def center_of_mass_position(pos:np.ndarray, weight:np.ndarray)-> float:
    """Compute the center of mass position given positions and weights.

    Args:
        pos (np.ndarray): Array of positions (e.g., pixel indices or wavelengths).
        weight (np.ndarray): Array of weights (e.g., intensity values).

    Returns:
        float: The center of mass position.
    """
    total_weight = np.nansum(weight)
    if total_weight == 0:
        return np.nan
    com = np.nansum(pos * weight) / total_weight
    return com

def generate_line_profile(fn:Path,win:str,line_range = Tuple[float, float], bg_range = Tuple[float, float], za_range: Tuple[float, float] = (-17,15), SAVE: bool = False ) -> xr.Dataset|bool:
    """Generate the line profile from the given l1a file and wavelength bounds.

    Args:
        fn (Path): Path to the l1a netCDF file. Note: should be a file all frames are taken during complete nighttime conditions for best results.
        line_range (Tuple[float, float]): Wavelength range for the emission line Tuple(min,max) in nm.
        bg_range (Tuple[float, float]): Wavelength range for the background Tuple(min,max) in nm.
        za_range (Tuple[float, float], optional): Wavelength range for the z-axis Tuple(min,max). Defaults to (-17,15) in degrees.
        SAVE (bool, optional): Whether to save the generated line profile. Defaults to False.

    Returns:
        Xr.Dataset|bool: countrate line profile as a function of za in degrees or True if saved.
    """
    if not isinstance(line_range, Tuple):
        raise ValueError("line_range must be a tuple of (min, max)")
    if not isinstance(bg_range, Tuple):
        raise ValueError("bg_range must be a tuple of (min, max)")

    ds = xr.open_dataset(fn)
    ids = ds.countrate.sum('tstamp').clip(min=0)

    tds = ids.sel(wavelength = slice(line_range[0], line_range[1])) #type: ignore
    bgds = ids.sel(wavelength = slice(bg_range[0], bg_range[1])) #type: ignore
    idx = np.min([tds.wavelength.shape[0], bgds.wavelength.shape[0]])
    tds = tds.isel(wavelength = slice(0, idx))
    bgds = bgds.isel(wavelength = slice(0, idx))
    tds.data = tds.data - bgds.data
    tds = tds.clip(min=0)

    norm = xr.apply_ufunc(
        center_of_mass_position,
        tds.wavelength, 
        tds,
        input_core_dims=[['wavelength'], ['wavelength']],
        vectorize=True)
    
    norm = norm.sel(za = slice(za_range[0], za_range[1]))
    norm -= (norm.max())
    norm = norm.to_dataset(name='line_profile')

    norm['line_profile'].attrs['description'] = f'Line profile (deviation from wavelength at mx intensity for each za) for window {win} extracted from {fn.name}'
    norm['line_profile'].attrs['units'] = 'nm'
    norm['za'].attrs = ds.za.attrs
    norm.attrs['source_file'] = str(fn)
    norm.attrs['line_range'] = f'{line_range[0]} nm to {line_range[1]} nm' #type: ignore
    norm.attrs['bg_range'] = f'{bg_range[0]} nm to {bg_range[1]} nm' #type: ignore
    norm.attrs['creation_date'] = pd.Timestamp.now().isoformat()

    if SAVE:
        outfn = f'line_profile_{win}.nc'
        norm.to_netcdf(outfn)
        print(f"Line profile saved to {outfn}")
        return True
    else:
        return norm 

#%%
if __name__ == "__main__":
    dir = Path('../../data/l1a')
    bounds_df = pd.read_csv(Path(__file__).parent / 'bounds.csv', comment='#')
    bounds_df.set_index('id', inplace=True) 

    for w in ['5577','6300']:   
        fn = list(dir.glob(f'*/*{w}*[0]*.nc'))
        line_bounds = (bounds_df.loc[f'{w}']['wlmin'], bounds_df.loc[f'{w}']['wlmax'])
        bg_bounds = (bounds_df.loc[f'{w}_bg']['wlmin'], bounds_df.loc[f'{w}_bg']['wlmax'])
        generate_line_profile(fn[0], w, line_range=line_bounds, bg_range=bg_bounds, SAVE=True)

    TEST = False

    if TEST: 
        win = '5577'
        ds = xr.open_dataset(f'line_profile_{win}.nc')
        ds.line_profile.plot(y='za')
        plt.axvline(0, color='r', ls='--')
        plt.title(f'Line profile for {int(win)/10:0.1f} nm emission line')
        plt.show()


# %%
