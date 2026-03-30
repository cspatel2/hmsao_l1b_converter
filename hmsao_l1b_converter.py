# Convert L1A -> L1B
# perform secondary straightening on a l1a dataset based on a provided line profile
# %%
import argparse
from datetime import datetime
from tkinter import NO
import numpy as np
import xarray as xr
from pathlib import Path
from secondary_straightening import secondary_straightening
from tqdm import tqdm
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter1d
import os
import sys
from matplotlib import pyplot as plt

LOCALPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(LOCALPATH))
from l1b_helpers import apply_flatfield_correction

#%%
@dataclass
class L1BConfig:
    """ Configuration for L1B conversion.
    Arguments:
        rootdir (str|Path): Root directory containing L1A files.
        destdir (str|Path): Destination directory for L1B files. If not provided, will create a 'l1b' directory parallel to the rootdir.
        windows (list[str]): List of windows to process. If not provided, will process all windows found in the calibration map directory.
        dates (list[str]): List of dates to process. If not provided, will process all dates found in the rootdir.
        flatdir (str|Path): Path to directory containing flat field files to apply flat field correction to the data before performing secondary straightening. This should be a .nc file containing a variable named 'countrate' with dimensions that match the data (e.g. za, wavelength). If not provided, no flat field correction will be applied.
        line_profile_dir (str|Path): Directory containing line profile (.nc) files. To generate them, use the line profile generator in the secondary_straightening module.
    """
    rootdir: str | Path
    destdir: str | Path
    windows: list[str]
    dates: list[str]
    flatdir: str | Path
    line_profile_dir: str | Path

def main(config: L1BConfig):
    """ Convert L1A -> L1B
        perform secondary straightening on a l1a dataset based on a provided line profile

        Arguments:
            config (L1BConfig): Configuration for L1B conversion.
            Inputs:
                - rootdir (str|Path): Root directory containing L1A files.
                - destdir (str|Path): Destination directory for L1B files.
                - windows (list[str]): List of windows to process. if windows is [''] or '', will process all windows found in the calibration map directory.
                - dates (list[str]): List of dates to process. if dates is [''] or '', will process all dates found in the rootdir.
                - flatdir (str|Path): Path to directory containing flat field files to apply flat field correction to the data before performing secondary straightening. This should be a .nc file containing a variable named 'countrate' with dimensions that match the data (e.g. za, wavelength). If not provided, no flat field correction will be applied.
                - line_profile_dir (str|Path): Directory containing line profile (.nc) files. To generate them, use the line profile generator in the secondary_straightening module.

    Raises:
        FileNotFoundError: if no line profile files found in the provided line_profile_dir
        ValueError: if multiple line profile files found for a window in the line_profile_dir
        ValueError: if no known data variable found in the L1A dataset (expects either 'countrate' or 'intensity')
    """    
    #check rootdir
    if isinstance(config.rootdir, str):
        config.rootdir = Path(config.rootdir)
    config.rootdir = config.rootdir.expanduser()

    #check Destination Dir
    if isinstance(config.destdir, str) and config.destdir == '':
        config.destdir = Path(str(config.rootdir).replace('l1a', 'l1b'))
    elif isinstance(config.destdir, str):
        config.destdir = Path(config.destdir)
    config.destdir = config.destdir.expanduser()
    if config.destdir.parts[-1] != 'l1b':
        config.destdir = config.destdir / 'l1b'
    config.destdir.mkdir(exist_ok=True, parents=True)
    print(f"Destination directory: {config.destdir}")

    #check dates
    if config.dates == '': # process all files
        config.dates = ['']

    if config.flatdir == '' or config.flatdir is not None: #check if flat field correction is needed
        config.flatdir = Path(config.flatdir).expanduser()
        if config.flatdir.is_file():
            raise ValueError(f"Provided flatdir {config.flatdir} is a file. Please provide a directory containing flat field .nc files with 'flat' in the filename.")
        if config.flatdir.is_dir() and config.flatdir.parts[-1] != 'l1a':
            config.flatdir = config.flatdir / 'l1a'
        flat_fns = list(config.flatdir.glob('*flat*.nc'))
        if len(flat_fns) < 1:
            raise FileNotFoundError(f"No flat field files found in {config.flatdir}. Please provide a valid directory containing flat field .nc files with 'flat' in the filename.")
    else:
        print("No flat field directory provided. Skipping flat field correction.")
    print(f"config.flatdir: {config.flatdir}")
    
    # check line profile dir
    if isinstance(config.line_profile_dir, str):
        config.line_profile_dir = Path(config.line_profile_dir)
    config.line_profile_dir = config.line_profile_dir.expanduser()
    lp_fnames = list(config.line_profile_dir.glob('*line_profile_*.nc'))
    if len(lp_fnames) < 1:
        raise FileNotFoundError(f"No line profile files found in {config.line_profile_dir}. Please provide a valid directory containing line profile .nc files.")   


    #windos to process
    available_windows = [f.stem.split('_')[-1] for f in lp_fnames]
    if config.windows == [''] or config.windows == '':
        valid_windows = available_windows
    else: valid_windows = list(set(config.windows) & set(available_windows))
    # print(f"Available windows in line profile directory: {available_windows}")
    print(f"Windows to be processed: {valid_windows}")

    for win in tqdm(valid_windows):
        print(f"Processing window: {win}")

        #get flat field
        flatds,darkds = None, None
        if config.flatdir is not None:
            flat_fns = np.sort(list(config.flatdir.glob(f'*flat*{win}*.nc')))[0] #type: ignore
            flatds = xr.open_dataset(flat_fns)
            #get dark to correct flat field if dark corrected data is being processed
            dark_fns = np.sort(list(config.flatdir.glob(f'*dark*{win}*.nc')))[0] #type: ignore
            darkds = xr.open_dataset(dark_fns)
        # print(f"flatds: {flatds}, \n\n darkds: {darkds}")
        
        #get line profile
        lp_file = list(config.line_profile_dir.glob(f'*line_profile_{win}*.nc'))
        if len(lp_file) > 1:
            raise ValueError(f"Multiple line profile files found for window {win} in {config.line_profile_dir}. Please ensure there is only one line profile file per window.")   
        else:
            lp_file = lp_file[0]
        lprof = xr.open_dataset(lp_file)
        lprof['line_profile'] = lprof['line_profile'] = (lprof.dims, gaussian_filter1d(lprof.line_profile.values, sigma=5)) #smooth line profile with gaussian filter to avoid overfitting to noise in the line profile

        for date in config.dates:
            #get L1A data files
            fns = list(config.rootdir.glob(f'*/*{date}*{win}*.nc'))
            print(f"Found {len(fns)} files to process.")
            fns.sort()

             #check main varirable in dataset
            ds = xr.open_dataset(fns[0])
            if 'countrate' in list(ds.data_vars):
                id = 'countrate'
            elif 'intensity' in list(ds.data_vars):
                id = 'intensity'
            else:
                raise ValueError("no known data variable found")
            
            #check if data is dark corrected and do that for flatfield
            if flatds is not None and darkds is not None:
                if ' not ' not in ds.Note.lower(): # check if ds is dark corrected.
                    flatds['countrate'] = flatds['countrate'] - darkds['countrate']
            
            for fn in tqdm(fns, desc=f"window {win} | date {date}:"):
                # print(f"Processing file: {fn.name}...", end='', flush=True)
                ds = xr.open_dataset(fn)
                if id == 'intensity':
                    ds = ds.rename({'intensity': 'countrate'})
                
                #apply flat field correction
                if flatds is not None:
                    da = ds['countrate']
                    da = apply_flatfield_correction(da, flatds['countrate'], win=win, in_place=True, PLOT=False)
                    ds['countrate'] = da

                ss = secondary_straightening(ds, lprof)
                ss['countrate'] = ss['countrate'].clip(min=0)
                all_vars = list(ds.coords) + list(ds.keys())
                for var in all_vars:
                    ss[var].attrs = ds[var].attrs
                ss.attrs['DataProcessingLevel'] = 'L1b - Secondary Straightened'
                ss.attrs['FileCreationDate'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S EDT")
                encoding = {var: {'zlib': True} for var in (*ds.data_vars.keys(), *ds.coords.keys())}
                # print('\tsaving...', end='', flush=True)
                outfn_dir = config.destdir / str(fn.parent).split('/l1a/')[-1]  # get the subdirs after l1a and replicate them in destdir
                outfn_dir.mkdir(parents=True, exist_ok=True) 
                outfn = outfn_dir/ fn.name.replace('l1a', 'l1b') #use same name but replace l1a with l1b
                try:
                    ss.to_netcdf(outfn, encoding=encoding)
                except Exception as e:
                    outfn.unlink(missing_ok=True) #delete file if it was created but an error occurred during saving to avoid leaving corrupted files
                ss.to_netcdf(outfn, encoding=encoding)
                # print('\tDone.', flush=True)

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert L1A datasets to L1B by performing secondary straightening based on provided line profiles.")
    parser.add_argument('--rootdir', type=str, required=True, help="Root directory containing L1A files.")
    parser.add_argument('--destdir', type=str, default='', help="Destination directory for L1B files. If not provided, will create a 'l1b' directory parallel to the rootdir.")
    parser.add_argument('--windows', nargs='+', default=[''], help="List of windows to process. If not provided, will process all windows found in the calibration map directory.")
    parser.add_argument('--dates', nargs='+', default=[''], help="List of dates to process. If not provided, will process all dates found in the rootdir.")
    parser.add_argument('--flatdir', type=str, default='', help="Path to directory containing flat field files to apply flat field correction to the data before performing secondary straightening. This should be a .nc file containing a variable named 'countrate' with dimensions that match the data (e.g. za, wavelength). If not provided, no flat field correction will be applied.")
    parser.add_argument('--line_profile_dir', type=str, required=True, help="Directory containing line profile (.nc) files. To generate them, use the line profile generator in the secondary_straightening module.")   

    args = parser.parse_args()
    config = L1BConfig(
        rootdir=args.rootdir,
        destdir=args.destdir,
        windows=args.windows,
        dates=args.dates,
        flatdir=args.flatdir,
        line_profile_dir=args.line_profile_dir
    ) #create config dataclass instance from parsed arguments
    main(config)
    
