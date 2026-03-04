#%%
#this file creates the line profile of a chosen emission line from a given l1a file, to use as input to the secondary straightening process
import argparse
from ast import List

from dataclasses import dataclass
import xarray as xr
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Iterable, List, SupportsFloat as Numeric
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

def generate_line_profile(fn:Path|str,win:str,line_range = Tuple[float, float], bg_range = Tuple[float, float], za_range: Tuple[float, float] = (-17,15), SAVE: Path|str = '' ) -> xr.Dataset|bool:
    """Generate the line profile from the given l1a file and wavelength bounds.

    Args:
        fn (Path|str): Path to the l1a netCDF file. Note: should be a file all frames are taken during complete nighttime conditions for best results.
        line_range (Tuple[float, float]): Wavelength range for the emission line Tuple(min,max) in nm.
        bg_range (Tuple[float, float]): Wavelength range for the background Tuple(min,max) in nm.
        za_range (Tuple[float, float], optional): Wavelength range for the z-axis Tuple(min,max). Defaults to (-17,15) in degrees.
        SAVE (Path|str, optional): Path to save the generated line profile. If empty string, no saving is performed.

    Returns:
        Xr.Dataset|bool: countrate line profile as a function of za in degrees or True if saved.
    """
    if not isinstance(line_range, Tuple):
        raise ValueError("line_range must be a tuple of (min, max)")
    if not isinstance(bg_range, Tuple):
        raise ValueError("bg_range must be a tuple of (min, max)")
    print('getting dataset...')
    if isinstance(fn, str): fn = Path(fn)
    ds = xr.open_dataset(fn)
    ids = ds.countrate.sum('tstamp').clip(min=0)

    tds = ids.sel(wavelength = slice(line_range[0], line_range[1])) #type: ignore
    bgds = ids.sel(wavelength = slice(bg_range[0], bg_range[1])) #type: ignore
    idx = np.min([tds.wavelength.shape[0], bgds.wavelength.shape[0]])
    tds = tds.isel(wavelength = slice(0, idx))
    bgds = bgds.isel(wavelength = slice(0, idx))
    tds.data = tds.data - bgds.data
    tds = tds.clip(min=0)
    
    print('calculating line profile...')
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

    
    if SAVE != '':        
        outdir = SAVE if isinstance(SAVE, Path) else Path(SAVE)
        outfn = outdir / f'line_profile_{win}.nc'
        print(f'saving line profile to {outfn}...')
        norm.to_netcdf(outfn)
        print(f"Line profile saved to {outfn}")
        return True
    else:
        return norm 


@dataclass
class LineProfileConfig:
    rootdir: str
    date:str #in yyyymmdd format, should match the date in the filename of the l1a file you want to use
    bounds_csv: str
    win:List[str]
    destdir: str

def main(config: LineProfileConfig):
    bounds_df = pd.read_csv(Path(config.bounds_csv), comment='#')
    bounds_df.set_index('id', inplace=True) 

    if config.win == [''] or config.win is None: #if no window specified, generate line profiles for all windows found in the bounds.csv
        valid_windows = list(bounds_df.index.str.replace('_bg','').unique())
    else: #check that specified windows are valid and generate line profiles for those in bounds.csv
        valid_windows = [w for w in config.win if f'{w}' in list(bounds_df.index.str.replace('_bg','').unique())]
    
    rootdir = Path(config.rootdir).expanduser()
    destdir = Path(config.destdir).expanduser()
    print(f"Destination directory for line profiles: {destdir}")
    # destdir.mkdir(parents=True, exist_ok=True)
    print('valid windows to generate line profiles for:', valid_windows)
    for w in valid_windows:
        print(f"Generating line profile for window {w} on date {config.date}...")   
        fn = list(rootdir.glob(f'*/*{config.date}*{w}*[0]*.nc'))
        if len(fn) == 0:
            print(f"No files found for window {w} on date {config.date}. Skipping.")
            continue
        print(f'getting bounds for window {w} from bounds.csv...')
        line_bounds = (bounds_df.loc[f'{w}']['wlmin'], bounds_df.loc[f'{w}']['wlmax'])
        bg_bounds = (bounds_df.loc[f'{w}_bg']['wlmin'], bounds_df.loc[f'{w}_bg']['wlmax'])
        generate_line_profile(fn[0], w, line_range=line_bounds, bg_range=bg_bounds, SAVE= destdir) #type: ignore

    return True



#%%
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate line profile for a given emission line from an l1a file to use as the starting profile for secondary straigtening.")
    parser.add_argument('--rootdir',
        type=str,
        required=True,
        nargs='?',
        help = 'Root directory for the data. Should be level 1a (.nc) files.')
    parser.add_argument('--date',
    type=str,
    required=True,
    nargs='?',
    help = 'Date of the data to use in YYYYMMDD format. Should match the date in the filename of the l1a file you want to use.')
    parser.add_argument('--bounds_csv',
        type=str,
        required=True,
        nargs='?',
        help = 'Path to the bounds.csv file containing wavelength bounds for the line and background regions for each window.')
    parser.add_argument('--win',
        type=str,
        required=False,
        default='',
        nargs='?',
        help = 'window to generate line profile for. Should match the window in the filename of the l1a file you want to use (e.g., 5577 or 6300). If '', it will generate line profiles for all windows found in the bounds.csv.')
    parser.add_argument('--destdir',
        type=str,
        required=True,
        nargs='?',
        help = 'Directory to save the generated line profile.')
    
    args = parser.parse_args()

    config = LineProfileConfig(
        rootdir=args.rootdir,
        date=args.date,
        bounds_csv=args.bounds_csv,
        win=args.win,
        destdir=args.destdir
    )

    main(config)

    TEST = False #set to True to run the test code below to generate line profiles for all windows in bounds.csv using the example data in this repo and plot the line profile for the 5577 window.
    if TEST:
        win = '5577'
        ds = xr.open_dataset(Path(config.destdir).expanduser() / f'line_profile_{win}.nc')
        ds.line_profile.plot(y='za')
        plt.axvline(0, color='r', ls='--')
        plt.title(f'Line profile for {int(win)/10:0.1f} nm emission line')
        plt.show()


    #%%

    # dir = Path('../../data/l1a')
    # bounds_df = pd.read_csv(Path(__file__).parent / 'bounds.csv', comment='#')
    # bounds_df.set_index('id', inplace=True) 

    # for w in ['5577','6300']:   
    #     fn = list(dir.glob(f'*/*{w}*[0]*.nc'))
    #     line_bounds = (bounds_df.loc[f'{w}']['wlmin'], bounds_df.loc[f'{w}']['wlmax'])
    #     bg_bounds = (bounds_df.loc[f'{w}_bg']['wlmin'], bounds_df.loc[f'{w}_bg']['wlmax'])
    #     generate_line_profile(fn[0], w, line_range=line_bounds, bg_range=bg_bounds, SAVE=Path(f'line_profile_{w}.nc'))

    # TEST = False

    # if TEST: 
    #     win = '5577'
    #     ds = xr.open_dataset(f'line_profile_{win}.nc')
    #     ds.line_profile.plot(y='za')
    #     plt.axvline(0, color='r', ls='--')
    #     plt.title(f'Line profile for {int(win)/10:0.1f} nm emission line')
    #     plt.show()


# %%
