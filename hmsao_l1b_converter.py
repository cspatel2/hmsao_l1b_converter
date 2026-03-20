# Convert L1A -> L1B
# perform secondary straightening on a l1a dataset based on a provided line profile
# %%
import argparse
from ast import arg
from datetime import datetime
from matplotlib.style import available
import numpy as np
import xarray as xr
from skimage import transform
from pathlib import Path
from secondary_straightening import secondary_straightening
from matplotlib import pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter1d


#%%
@dataclass
class L1BConfig:
    """ Configuration for L1B conversion.
    Arguments:
        rootdir (str|Path): Root directory containing L1A files.
        destdir (str|Path): Destination directory for L1B files. If not provided, will create a 'l1b' directory parallel to the rootdir.
        windows (list[str]): List of windows to process. If not provided, will process all windows found in the calibration map directory.
        line_profile_dir (str|Path): Directory containing line profile (.nc) files. To generate them, use the line profile generator in the secondary_straightening module.
    """
    rootdir: str | Path
    destdir: str | Path
    windows: list[str]
    dates: list[str]
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

    for win in valid_windows:
        print(f"Processing window: {win}")

        #get line profile
        lp_file = list(config.line_profile_dir.glob(f'*line_profile_{win}*.nc'))
        if len(lp_file) > 1:
            raise ValueError(f"Multiple line profile files found for window {win} in {config.line_profile_dir}. Please ensure there is only one line profile file per window.")   
        else:
            lp_file = lp_file[0]
        lprof = xr.open_dataset(lp_file)
        lprof['line_profile'] = lprof['line_profile'] = (lprof.dims, gaussian_filter1d(lprof.line_profile.values, sigma=5)) #smooth line profile with gaussian filter to avoid overfitting to noise in the line profile

        
        print(f"Processing window: {win}")


        
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
            
            for fn in tqdm(fns, desc=f"window {win} | date {date}:"):
                # print(f"Processing file: {fn.name}...", end='', flush=True)
                ds = xr.open_dataset(fn)
                if id == 'intensity':
                    ds = ds.rename({'intensity': 'countrate'})
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
                ss.to_netcdf(outfn, encoding=encoding)
                # print('\tDone.', flush=True)

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert L1A datasets to L1B by performing secondary straightening based on provided line profiles.")
    parser.add_argument('--rootdir', type=str, required=True, help="Root directory containing L1A files.")
    parser.add_argument('--destdir', type=str, default='', help="Destination directory for L1B files. If not provided, will create a 'l1b' directory parallel to the rootdir.")
    parser.add_argument('--windows', nargs='+', default=[''], help="List of windows to process. If not provided, will process all windows found in the calibration map directory.")
    parser.add_argument('--dates', nargs='+', default=[''], help="List of dates to process. If not provided, will process all dates found in the rootdir.")
    parser.add_argument('--line_profile_dir', type=str, required=True, help="Directory containing line profile (.nc) files. To generate them, use the line profile generator in the secondary_straightening module.")   

    args = parser.parse_args()
    config = L1BConfig(
        rootdir=args.rootdir,
        destdir=args.destdir,
        windows=args.windows,
        dates=args.dates,
        line_profile_dir=args.line_profile_dir
    )
    main(config)
    



        
        
        

# %%
