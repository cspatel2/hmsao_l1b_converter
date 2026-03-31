#%%
#this file creates the line profile of a chosen emission line from a given l1a file, to use as input to the secondary straightening process
import argparse
from curses import window
from dataclasses import dataclass
from collections.abc import Callable
import xarray as xr
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from typing import Dict, Iterable, List, SupportsFloat as Numeric
from scipy.signal import find_peaks
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from l1b_helpers import apply_flatfield_correction
#%%

def Gaussian(x: np.ndarray, a: float, xo: float, sigma: float, r: float) -> np.ndarray:
    """ Guassian function

    Args:
        x (np.ndarray): x values (e.g., wavelength)
        a (float): amplitude of the Gaussian
        xo (float): center of the Gaussian (e.g., line center wavelength)
        sigma (float): standard deviation of the Gaussian (e.g., line width)
        r (float): constant offset (e.g., background level)

    Returns:
        np.ndarray: y values (e.g., countrate)
    """    
    return a * np.exp(-((x - xo) ** 2) / (2 * sigma**2)) + r

def rsquared_for_func(func: Callable, x: np.ndarray, y: np.ndarray, popt: tuple) -> float:
    """ Calculate Coeff of determination (R-squared) for custom function. 

    Args:
        func (Callable): The function for which to calculate R-squared (e.g., Gaussian)
        x (np.ndarray): x values (e.g., wavelength)
        y (np.ndarray): y values (e.g., countrate)
        popt (tuple): Optimal parameters for the function (e.g., Gaussian fit parameters)

    Returns:
        float: R-squared value indicating goodness of fit (1 is perfect fit, 0 means no fit)
    """    
    residuals = y - func(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

def Normalize(da: xr.DataArray, dim: str | None = None, invert_signal: bool = False) -> xr.DataArray:
    """ Normalize an xarray DataArray either globally or along a specified dimension, with optional signal inversion.

    Args:
        da (xr.DataArray): The DataArray to normalize.
        dim (str | None, optional): The dimension along which to normalize. If None, normalize globally. Defaults to None.
        invert_signal (bool, optional): Whether to invert the signal (e.g., for daytime data). Defaults to False.

    Returns:
        xr.DataArray: The normalized (and optionally inverted) DataArray.
    """
    if dim is None:
        # normalize the entire array globally
        da = da - da.min()
        da = da / da.max()
    else:
        # normalize each slice along `dim` independently
        da = da - da.min(dim=dim)
        da = da / da.max(dim=dim)
    if invert_signal:
        da = 1 - da
    return da

def determine_wl_to_track(
    ds: xr.Dataset | xr.DataArray,func:Callable, invert_signal: bool = False, PLOT=False
) -> Tuple[float, float]:
    """ Determine the wavelength to track for line profile generation by finding peaks in the signal and evaluating their "Gaussian-ness".

    Args:
        ds (xr.Dataset | xr.DataArray): The dataset or data array containing the signal.
        func (Callable): The function to use for curve fitting of spectral line.
        invert_signal (bool, optional): Whether to invert the signal (e.g., set to True for daytime [absorption] data). Defaults to False.
        PLOT (bool, optional): Whether to plot the results. Defaults to False.

    Returns:
        Tuple[float, float]: The central wavelength to track for line profile generation and the window size for the line profile.
    """

    if isinstance(ds, xr.Dataset):
        da = ds.countrate
    else:
        da = ds

    if "tstamp" in da.dims:
        da = da.mean("tstamp")
    if "za" in da.dims:
        da = da.sum("za")

    # nnormalize
    da = Normalize(da, invert_signal=invert_signal)
    # find peaks in the signal and return the corresponding wavelengths
    peaks, _ = find_peaks(da.values, height=0.4, distance=35)
    # test the "Guassian-ness" of each peak
    scores, sigmas = [], []
    windowsize = 0.25
    for p in peaks:
        tda = da.sel(
            wavelength=slice(
                da.wavelength.values[p] - windowsize,
                da.wavelength.values[p] + windowsize,
            )
        )
        fitda = tda.curvefit(
            coords="wavelength",
            func=func,
            skipna=True,
            p0={
                "a": np.abs(
                    np.abs(tda.max(skipna=True)) - np.abs(tda.min(skipna=True))
                ),
                "xo": da.wavelength.values[p],
                "sigma": windowsize / 2,
                "r": 0,
            },errors="ignore",)  # type: ignore
        sigmas.append(fitda.curvefit_coefficients.sel(param="sigma").values)
        score = rsquared_for_func(func, tda.wavelength.values, tda.values, fitda.curvefit_coefficients.values)
        scores.append(score)
    #use scores to determine the best peak and return the corresponding wavelength
    best_peak_idx = np.argmax(scores)
    best_peak = da.wavelength.values[peaks[best_peak_idx]]
    best_windowsize = sigmas[best_peak_idx]
    if PLOT:
        plt.figure()
        da.plot(label="Signal")  # type: ignore
        plt.scatter(da.wavelength.values[peaks], da.values[peaks], color="r", label="Peaks")  # type: ignore
        plt.axvline(
            best_peak, color="g", ls="--", label=f"Best Peak: {best_peak.values:.2f} nm"
        )
        plt.legend(loc="best")
    return best_peak, best_windowsize


def estimate_line_profile_curvefit(da: xr.DataArray, central_wl: float, windowsize: float, func: Callable, invert_signal: bool = False, plot_idx: int | None = None) -> xr.Dataset:
    """ Estimate the line profile by fitting a curve (e.g., Gaussian) to the signal around the central wavelength for each za.

    Args:
        da (xr.DataArray): The DataArray containing the signal to fit.
        central_wl (float): The central wavelength to fit around.
        windowsize (float): The window size around the central wavelength to use for fitting.
        func (Callable): The function to use for curve fitting of spectral line.
        invert_signal (bool, optional): Whether to invert the signal (e.g., set to True for daytime [absorption] data). Defaults to False.
        plot_idx (int | None, optional): The index of the za value to plot for curve fitting. Defaults to None (no plot).

    Returns:
        xr.Dataset: Dataset containing the curve fit coefficients for each za.
    """

    da = Normalize(da, dim="wavelength", invert_signal=invert_signal)
    tda = da.sel(wavelength=slice(central_wl - windowsize, central_wl + windowsize))
    fitda = tda.curvefit(
        coords="wavelength",
        func=func,
        skipna=True,
        p0={
            "a": np.abs(np.abs(tda.max(skipna=True)) - np.abs(tda.min(skipna=True))),
            "xo": central_wl,
            "sigma": windowsize / 2,
            "r": 0,},errors="ignore",)  # type: ignore
    if plot_idx is not None:
        plt.figure()
        tda.isel(za=plot_idx).plot(label="Signal")  # type: ignore
        plt.plot(tda.wavelength.values, func(tda.wavelength.values, *fitda.curvefit_coefficients.isel(za=plot_idx).values), color="r", label="Gaussian Fit")  # type: ignore
        plt.legend(loc="best")
        plt.title(
            f"Estimating Line Profile \n Gaussian Fit for za = {tda.za.values[plot_idx]:.1f} degrees"
        )
        plt.savefig(
            f"gaussian_fit_za_{tda.za.values[plot_idx]:.1f}_{central_wl}.png", dpi=300
        )
    return fitda


def generate_line_profile(
    da: xr.DataArray,
    win: str,
    func: Callable = Gaussian,
    invert_signal: bool = False,
    zaslice: slice = slice(-17, 15),
    wlslice: slice | None = None,
    wl_window_size_nm: float = 0.15,
    plot_za_idx: int | None = None,
    PLOT: bool = False,
) -> xr.Dataset:
    """ Generate the line profile from the given xr.dataarray (preferably for data that has been primarily straightened but not secondarily straightened yet).

    Args:
        da (xr.DataArray): DataArray containing the image to analyze for line profile generation. Has to have the shape (za, wavelength) or (tstamp, za, wavelength) and contain a variable named 'wavelength' with the wavelength values for each pixel.
        flatda (xr.DataArray): DataArray containing the flat-field image for normalization.
        win (str): The window identifier for the line profile generation.
        func (Callable, optional): The function to use for curve fitting. Defaults to Gaussian.
        invert_signal (bool, optional): Whether to invert the signal. If signal is daytime-set this to True and if night-time set this to False. Defaults to False.
        zaslice (slice, optional): The slice of za values to analyze. Defaults to slice(-17, 15).
        wlslice (slice | None, optional): The slice of wavelength values to analyze. Defaults to None.
        wl_window_size_nm (float, optional): The window size for the wavelength slice to analyze around the central wavelength (~the line width of the spectral line) in nm. Defaults to 0.15.
        plot_za_idx (int | None, optional): The index of the za value to plot for curve fitting. Defaults to None (No plot).
        PLOT (bool, optional): plot the results. Defaults to False.

    Returns:
        xr.Dataset: _description_
    """    
    if "tstamp" in da.dims:
        da = da.mean("tstamp")
    if "za" in da.dims:
        da = da.sel(za=zaslice)
    if wlslice is None:
        wlslice = slice(da.wavelength.min() + 1.5, da.wavelength.max() - 0.9)
    da = da.sel(wavelength=wlslice)

    # Find the wavelength of the spectral line to track
    central_wl, _ = determine_wl_to_track(da, func=func, invert_signal=invert_signal, PLOT=PLOT)
    # Estimate the line profile by curve fitting using the central wavelength determined above
    curvefit = estimate_line_profile_curvefit(da, central_wl, wl_window_size_nm, func=func, invert_signal=invert_signal, plot_idx=plot_za_idx)

    # fit a polynomial to the line profile to smooth it out 
    lp = curvefit.curvefit_coefficients.sel(param="xo")
    mask = (lp > lp.quantile(0.02)) & (lp < lp.quantile(0.98))
    lp = lp.where(mask)
    ft = lp.polyfit("za", deg=3)
    fitted_line_cf = xr.polyval(lp.za, ft.polyfit_coefficients)

    # normalize the line profile 
    dist_to_max = np.abs(fitted_line_cf.max().values - int(win) / 10)
    dist_to_min = np.abs(fitted_line_cf.min().values - int(win) / 10)
    normalize_to_idx = np.argmax([dist_to_max, dist_to_min])
    #takes care of the curve direction being flipped for top windows vs bottom windows, by checking whether the fitted line profile is closer to the expected wavelength at the max or min and normalizing accordingly
    if normalize_to_idx == 0:
        normalize_to_wl = fitted_line_cf.max().values
    else:
        normalize_to_wl = fitted_line_cf.min().values
    norm_fitted_line_cf = fitted_line_cf - normalize_to_wl
    #create dataset
    norm_fitted_line_cf.name = "line_profile"
    norm_fitted_line_cf = norm_fitted_line_cf.drop_vars("param")
    lpds = norm_fitted_line_cf.to_dataset()

    lpds.attrs["description"] = (
        f"Line profile (deviation from wavelength at mx intensity for each za) for window {win}."
    )
    lpds["line_profile"].attrs["long_name"] = f"Normalized Line profile"
    lpds["line_profile"].attrs["units"] = "nm"
    lpds["za"].attrs = da.za.attrs
    # norm_fitted_line_cf.attrs['source_file'] = str(fn)
    lpds.attrs["FileCreationDate"] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S EDT")

    if PLOT:
        plt.figure()
        lp.plot(y="za", label="Curve Fit Line Profile")  # type: ignore
        fitted_line_cf.plot(y="za", label="Fitted Line Profile")  # type: ignore
        plt.legend(loc="best")
        plt.title(f"Line Profile for window {win}")
        plt.show()
        plt.savefig(f"line_profile_{win}.png", dpi=300)

    return lpds


@dataclass
class LineProfileConfig:
    rootdir: str | Path
    flatdir: str | Path
    destdir: str | Path 
    date:str | list[str]#in yyyymmdd format, should match the date in the filename of the l1a file you want to use
    overwrite: bool = False

def main(config: LineProfileConfig):

    if isinstance(config.rootdir,str):
        config.rootdir = Path(config.rootdir)
    config.rootdir = config.rootdir.expanduser()

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

    if isinstance(config.destdir,str):
        config.destdir = Path(config.destdir)
    config.destdir = config.destdir.expanduser()
    config.destdir.mkdir(parents=True, exist_ok=True)

    print(list(config.flatdir.glob('l1a/*flat*.nc')))
    available_windows = np.unique([a.stem.split('_')[-1] for a in list(config.flatdir.glob(f'*flat*.nc'))])

    SPECTRA_TYPE_PARAMS = {
        'emission': {
            'invert_signal': False, #nighttime 
            'fidx': 0,
        },
        'adsorption': {
            'invert_signal': True, #daytime
            'fidx': 4,
        }
    }

    PARAMS = {
        '5577': {
            'type': 'emission',
            'windowsize': 0.15,
        },
        '6300': {
            'type': 'emission',
            'windowsize': 0.15,
        },
        '6563': {
            'type': 'adsorption',
            'windowsize': 0.2,
        },
        '4861': {
            'type': 'adsorption',
            'windowsize': 0.2,
        },
        '7774': {
            'type': 'adsorption',
            'windowsize': 0.2,
        },
    }
    print('available_windows: ', available_windows)
    for win in available_windows:
        if win not in PARAMS.keys():
            print(f"Window {win} not found in PARAMS dictionary. Skipping line profile generation for this window. Please add the necessary parameters for window {win} to the PARAMS dictionary to generate line profiles for this window.")
            continue
        spectra_type = PARAMS[win]['type']
        fidx = SPECTRA_TYPE_PARAMS[spectra_type]['fidx']
        invert_signal = SPECTRA_TYPE_PARAMS[spectra_type]['invert_signal']
        windowsize = PARAMS[win]['windowsize']
    

        #get data
        fns = list(config.rootdir.glob(f'**/*{config.date}*{win}*.nc'))
        if len(fns) < 1:
            continue
        fns.sort()
        ds = xr.open_dataset(fns[fidx])

        #get flat field data
        flatfn = list(config.flatdir.glob(f'*flat*{win}*.nc'))
        if len(flatfn) < 1:
            raise FileNotFoundError(f"No flat field file found for window {win} in {config.flatdir}. Please provide a valid directory containing flat field .nc files with 'flat' in the filename.")
        flatds = xr.open_dataset(flatfn[0])

        #apply flat field correction
        ds['countrate'] = apply_flatfield_correction(ds['countrate'], flatds['countrate'], win=win, in_place=True, PLOT=False)

        #generate line profile
        lpds = generate_line_profile(
            da=ds['countrate'],
            win=win,
            func=Gaussian,
            invert_signal=invert_signal,
            zaslice=slice(-17, 15),
            wlslice=None,
            wl_window_size_nm=windowsize,
            plot_za_idx=0,
            PLOT=False,
        )

        lpds.attrs['source_file'] = str(fns[fidx].name)
        lpds.attrs['window'] = win

        encoding = {var: {'zlib': True} for var in (*lpds.data_vars.keys(), *lpds.coords.keys())}

        #save line profile
        outfn = config.destdir / f'line_profile_{win}.nc'
        if config.overwrite and outfn.exists():
            outfn.unlink()
        lpds.to_netcdf(outfn, encoding=encoding)
        print(f"Line profile for window {win} saved to {outfn}")


#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate line profiles for HMSAO data.")
    parser.add_argument("--rootdir", type=str, required=True, help="Root directory containing the l1a files to process.")
    parser.add_argument("--flatdir", type=str, default='', help="Directory containing flat field .nc files with 'flat' in the filename for flat field correction. If not provided, flat field correction will be skipped.")
    parser.add_argument("--destdir", type=str, required=True, help="Destination directory to save the generated line profile .nc files.")
    parser.add_argument("--date", type=str, required=True, help="Date in yyyymmdd format to match in the filename of the l1a file to use for line profile generation.")
    parser.add_argument("--overwrite", action='store_true', help="Whether to overwrite existing line profile files in the destination directory. If not set, existing files will not be overwritten.")
    args = parser.parse_args()

    config = LineProfileConfig(
        rootdir=args.rootdir,
        flatdir=args.flatdir,
        destdir=args.destdir,
        date=args.date,
        overwrite=args.overwrite,
    )

    main(config)