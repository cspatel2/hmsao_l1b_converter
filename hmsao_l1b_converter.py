# perform secondary straightening on a l1a dataset based on a provided line profile
# %%
from datetime import datetime
import numpy as np
import xarray as xr
from skimage import transform
from pathlib import Path
from secondary_straightening import secondary_straightening
from matplotlib import pyplot as plt
from tqdm import tqdm

LOCALPATH = Path(__file__).parent

LINE_PROFILES = list(
    (LOCALPATH / 'secondary_straightening').glob('line_profile_*.nc')
)
WINDOWS = [fp.stem.split('_')[-1] for fp in LINE_PROFILES]


def main(datadir: Path = Path('/home/charmi/locsststor/proc/hmsao/l1a'), destdir: str | Path | None = ''):
    for (win, lp) in zip(WINDOWS, LINE_PROFILES):
        print(f"Processing window: {win}")
        if destdir is None:
            destdirp = Path('./l1b')
        elif isinstance(destdir, str) and destdir == '':
            destdirp = Path(str(datadir).replace('l1a', 'l1b'))
        elif isinstance(destdir, str):
            destdirp = Path(destdir)
        else:
            destdirp = destdir
        destdirp.mkdir(exist_ok=True)
        print(f"Destination directory: {destdirp}")
        fns = list(datadir.glob(f'*/*{win}*.nc'))
        print(f"Found {len(fns)} files to process.")
        fns.sort()

        if lp.exists():
            lprof = xr.open_dataset(lp)
        else:
            raise FileNotFoundError(f"Line profile file {lp} not found.")

        ds = xr.open_dataset(fns[0])
        if 'countrate' in list(ds.data_vars):
            id = 'countrate'
        elif 'intensity' in list(ds.data_vars):
            id = 'intensity'
        else:
            print("no known data variable found")
            continue

        for fn in fns:
            print(f"Processing file: {fn.name}...", end='', flush=True)
            ds = xr.open_dataset(fn)
            if id == 'intensity':
                ds = ds.rename({'intensity': 'countrate'})
            ss = secondary_straightening(ds, lprof)
            ss['countrate'] = ss['countrate'].clip(min=0)
            all_vars = list(ds.coords) + list(ds.keys())
            for var in all_vars:
                ss[var].attrs = ds[var].attrs
            ss.attrs['DataProcessingLevel'] = 'L1b - Secondary Straightened'
            ss.attrs['FileCreationDate'] = datetime.now().strftime(
                "%m/%d/%Y, %H:%M:%S EDT")
            encoding = {var: {'zlib': True}
                        for var in (*ds.data_vars.keys(), *ds.coords.keys())}
            print('\tsaving...', end='', flush=True)
            outfn = destdirp.joinpath(fn.stem.replace(
                'hmso-aorigin', 'hmsao_l1b') + fn.suffix)
            ss.to_netcdf(outfn, encoding=encoding)
            print('\tDone.', flush=True)

# %%
