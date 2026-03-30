#%%
import csv
import numpy as np
import xarray as xr
from typing import Iterable



def get_bounds_from_csv(csv_path: str, wavelength: str):
    """
    Get the bounds from the csv file.
    """
    with open(csv_path) as f:
        bg_dict = {}
        feature_dict = {}
        za_dict = {}
        for row in csv.reader(f):
            key = row[0]
            if key.startswith('#'):
                continue
            elif key.startswith(wavelength) and 'bg' in key:
                bg_dict[key] = slice(float(row[1]), float(row[2]))
            elif key.startswith(wavelength) and 'za' in key:
                za_dict[key] = slice(float(row[1]), float(row[2]))
            elif key.startswith(wavelength):
                feature_dict[key] = slice(float(row[1]), float(row[2]))
        if len(bg_dict) == 0 and len(feature_dict) == 0:
            raise ValueError(f'No bounds found for {wavelength} in {csv_path}')
    return feature_dict, bg_dict, za_dict


def str2bool(value: str) -> bool:
    if value.lower() in ('true', '1', 't', 'y', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'f', 'n', 'no'):
        return False
    raise ValueError("Invalid boolean value: {}".format(value))


def rms_func(data, axis=None):
    return np.sqrt(np.sum(data**2, axis=axis))

def align_spectra_core(sol: np.ndarray, sky: np.ndarray, wavelength: np.ndarray, offset: float = None) -> np.ndarray:
    sol_ = np.nan_to_num(sol)
    sky_ = np.nan_to_num(sky)

    if offset is None: #automatically find a lag by using scipy.signal.correlate
        sky_norm = (sky_ - np.mean(sky_)) / np.std(sky_)
        sol_norm = (sol_ - np.mean(sol_)) / np.std(sol_)
        corr = correlate(sky_norm, sol_norm, mode='full')
        lag = np.arange(-len(sol_) + 1, len(sky_))[np.argmax(corr)]
        offset = -lag * np.mean(np.diff(wavelength))

    shifted_wl = wavelength - offset
    interp_sol = np.interp(wavelength, shifted_wl, sol_, left=np.nan, right=np.nan)
    return interp_sol

def apply_alignment(sol_ds: xr.DataArray, sky_ds: xr.DataArray, offset:float = None) -> xr.DataArray:
    wl = sky_ds["wavelength"]
    # sky_ds = sky_ds.drop_vars('tstamp')
    # sol_ds = sol_ds.drop_vars('tstamp')

    # sky_ds,sol_ds = xr.broadcast(sky_ds, sol_ds)
    aligned = xr.apply_ufunc(
        align_spectra_core,
        sol_ds['intensity'],
        sky_ds['intensity'],
        wl,
        input_core_dims=[["wavelength"], ["wavelength"], ["wavelength"]],
        output_core_dims=[["wavelength"]],
        vectorize=True,
        dask="parallelized",
        kwargs={"offset": offset},
        on_missing_core_dim="copy",
        dask_gufunc_kwargs = {'allow_rechunk':True}
        

    )
    return aligned

def Residual(sf:int,skyspec:Iterable[float],solspec:Iterable[float])-> Iterable[float]:return solspec - sf*skyspec # type: ignore


def solar_subtraction_core(sky:np.ndarray, sol:np.ndarray)->np.ndarray:
    # finite = np.isfinite(sky) & np.isfinite(sol)
    # res = least_squares(Residual, 1, args = (sky[finite],sol[finite]), loss= 'soft_l1')
    # sf = res.x[0]
    mask = np.isfinite(sky) & np.isfinite(sol)
    if not np.any(mask):
        return np.full_like(sky, np.nan)

    sky_masked = sky[mask]
    sol_masked = sol[mask]

    denom = np.dot(sky_masked, sky_masked)
    if denom == 0:
        sf = 0
    else:
        sf = np.dot(sky_masked, sol_masked) / denom
    return sky - sf * sol

    # return sol - sf * sky


def apply_solar_subtraction(sol_ds: xr.Dataset, sky_ds: xr.Dataset):

    if isinstance(sol_ds, xr.Dataset):
        sol_ds = sol_ds['intensity']
    
    if isinstance(sky_ds, xr.Dataset):
        sky_ds = sky_ds['intensity']
 
    # if 'tstamp' in list(sol_ds.sizes.keys()):
    #     sol_ds['tstamp'] = sky_ds.tstamp
    # else:
    sol_ds = sol_ds.expand_dims(tstamp = sky_ds.tstamp.values)
    # sol_ds['tstamp'] = sky_ds.tstamp
    
    

    subtracted = xr.apply_ufunc(
        solar_subtraction_core,
        sol_ds,
        sky_ds,
        input_core_dims=[["wavelength"], ["wavelength"]],
        output_core_dims=[["wavelength"]],
        vectorize=True,
        dask="parallelized",
        on_missing_core_dim="copy",
        dask_gufunc_kwargs = {'allow_rechunk':True}
    )

    return subtracted

def find_outlier_pixels(data, tolerance=3, worry_about_edges=True):
    # This function finds the hot or dead pixels in a 2D dataset.
    # tolerance is the number of standard deviations used to cutoff the hot pixels
    # If you want to ignore the edges and greatly speed up the code, then set
    # worry_about_edges to False.
    #
    # The function returns a list of hot pixels and also an image with with hot pixels removed

    from scipy.ndimage import median_filter
    blurred = median_filter(data, size=2)
    difference = data - blurred
    threshold = tolerance*np.std(difference)

    # find the hot pixels, but ignore the edges
    hot_pixels = np.nonzero((np.abs(difference[1:-1, 1:-1]) > threshold))
    # because we ignored the first row and first column
    hot_pixels = np.array(hot_pixels) + 1

    # This is the image with the hot pixels removed
    fixed_image = np.copy(data)
    for y, x in zip(hot_pixels[0], hot_pixels[1]):
        fixed_image[y, x] = blurred[y, x]

    if worry_about_edges == True:
        height, width = np.shape(data)

        ### Now get the pixels on the edges (but not the corners)###

        # left and right sides
        for index in range(1, height-1):
            # left side:
            med = np.median(data[index-1:index+2, 0:2])
            diff = np.abs(data[index, 0] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[index], [0]]))
                fixed_image[index, 0] = med

            # right side:
            med = np.median(data[index-1:index+2, -2:])
            diff = np.abs(data[index, -1] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[index], [width-1]]))
                fixed_image[index, -1] = med

        # Then the top and bottom
        for index in range(1, width-1):
            # bottom:
            med = np.median(data[0:2, index-1:index+2])
            diff = np.abs(data[0, index] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[0], [index]]))
                fixed_image[0, index] = med

            # top:
            med = np.median(data[-2:, index-1:index+2])
            diff = np.abs(data[-1, index] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[height-1], [index]]))
                fixed_image[-1, index] = med
        ### Then the corners###

        # bottom left
        med = np.median(data[0:2, 0:2])
        diff = np.abs(data[0, 0] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[0], [0]]))
            fixed_image[0, 0] = med

        # bottom right
        med = np.median(data[0:2, -2:])
        diff = np.abs(data[0, -1] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[0], [width-1]]))
            fixed_image[0, -1] = med

        # top left
        med = np.median(data[-2:, 0:2])
        diff = np.abs(data[-1, 0] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[height-1], [0]]))
            fixed_image[-1, 0] = med

        # top right
        med = np.median(data[-2:, -2:])
        diff = np.abs(data[-1, -1] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[height-1], [width-1]]))
            fixed_image[-1, -1] = med

    return hot_pixels, fixed_image

import matplotlib.pyplot as plt
from matplotlib.contour import QuadContourSet
def get_boundary_from_contourlevel(art: QuadContourSet, level_idx: int, wlbins: np.ndarray)-> tuple[np.ndarray, np.ndarray]:
    """ estimate median za boundary using user chosen contour level.

    Args:
        art (QuadContourSet): contour of the image to find boundary for. this can be produced using ds.plot.contour(...)
        level_idx (int): index of the contour level to use for finding the boundary. this is the index of the levels in the QuadContourSet, not the actual value of the level.
        wlbins (np.ndarray): wavelength bins to use for finding the boundary. the function will find the max and min za for each wavelength bin.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the top and bottom za boundaries for each wavelength bin.
    """    
    segs = art.allsegs[level_idx]
    if not segs:
        return np.full(len(wlbins), np.nan), np.full(len(wlbins), np.nan)
    #get all the x,y vertices of the contour segments and stack them into a single array
    all_verts = np.vstack(segs)
    wl_verts = all_verts[:, 0]
    za_verts = all_verts[:, 1]
    
    top_za = np.full(len(wlbins), np.nan)
    bot_za = np.full(len(wlbins), np.nan)
    dw = wlbins[1] - wlbins[0]
    
    #find the max and min za for each wl bin by finding the vertices that are within dw of the wl bin center
    for i, w in enumerate(wlbins):
        mask = np.abs(wl_verts - w) < dw
        if mask.sum() > 0:
            top_za[i] = za_verts[mask].max()
            bot_za[i] = za_verts[mask].min()
    
    return top_za, bot_za

def get_za_boundaries(da: xr.DataArray, level_idx: tuple[int, int] = (1,1))-> tuple[float, float]:
    """ estimate za boundaries where there is a sudden drop in intensity.

    Args:
        da (xr.DataArray): data array to find boundaries for. this should be a 2D array with dimensions 'wavelength' and 'za'.
        level_idx (tuple[int, int]): index of the top and bottom contour levels to use for finding the boundaries. this is the index of the levels in the QuadContourSet, not the actual value of the level.
    
    Returns:
        tuple[float, float]: A tuple containing the top and bottom za boundaries.
    """    
    wlbins = da.wavelength.values
    art = da.plot.contourf(x='wavelength', y='za', levels=20)
    plt.close() # close the plot to avoid displaying it in the notebook
    top_idx, bot_idx = level_idx
    if top_idx == bot_idx: #if same contour level is used, calc just once.
        top_za, bot_za = get_boundary_from_contourlevel(art, level_idx=top_idx, wlbins=wlbins)
    else: # if different contour levels are used, calc separately for top and bottom boundaries.
        top_za, _ = get_boundary_from_contourlevel(art, level_idx=top_idx, wlbins=wlbins)
        _, bot_za = get_boundary_from_contourlevel(art, level_idx=bot_idx, wlbins=wlbins)

    top_val = np.nanmedian(top_za)
    bot_val = np.nanmedian(bot_za)

    return top_val, bot_val

def apply_flatfield_correction(da: xr.DataArray, flatda: xr.DataArray, win: str,in_place:bool = True,  PLOT : bool = False)-> xr.DataArray:

    input_da = da.copy()

    #estimate za boundaries using contour levels
    if win in ['6563', '4861']: level_idx = (3,1)
    else: level_idx = (1,10)    
    za_top, za_bot = get_za_boundaries(flatda, level_idx=level_idx)
    zaslice = slice(za_bot, za_top)

    #normalization factor for flat
    med = flatda.sel(za=zaslice).median(skipna=True).values 

    
    if not in_place:# keep same shape as input da.
        mask = (da.za >= za_bot) & (da.za <= za_top)
        #nan for outside of zabounds.
        flatda = flatda.where(mask, np.nan, drop=False)
        da = da.where(mask, np.nan, drop=False)
    else: # crop array to za boundaries. shape of array may be different.
        flatda = flatda.sel(za=zaslice)
        da = da.sel(za=zaslice)
    #apply flat field correction
    da = da / (flatda / med)

    if PLOT:
        tidx = 0
        zaidx = len(da.za) // 2
        widx = len(da.wavelength) // 2
        fig,ax = plt.subplots(2,2, figsize = (12,12))
        input_da.isel(tstamp=tidx).plot(ax=ax[0,0], vmin = 0)
        ax[0,0].axhline(za_bot, color = 'r', ls = '--', label = 'Estimated za boundaries')
        ax[0,0].axhline(za_top, color = 'r', ls = '--')
        ax[0,0].legend()
        ax[0,0].set_title('Raw')

        da.isel(tstamp=tidx).plot(ax=ax[0,1], vmin = 0)
        ax[0,1].axhline(da.za.values[zaidx], color = 'k', ls = '--')
        ax[0,1].axvline(da.wavelength.values[widx], color = 'k', ls = '--')
        ax[0,1].set_title('Corrected with median flat')

        input_da.isel(tstamp = tidx, za = zaidx).plot(ax = ax[1,0], x = 'wavelength', label = 'Raw')
        da.isel(tstamp = tidx, za = zaidx).plot(ax = ax[1,0], x = 'wavelength', label = 'Corrected with median flat')
        ax[1,0].legend()

        input_da.isel(tstamp = tidx, wavelength = widx).plot(ax = ax[1,1], y = 'za', label = 'Raw')
        da.isel(tstamp = tidx, wavelength = widx).plot(ax = ax[1,1], y = 'za', label = 'Corrected with median flat')

        ax[1,1].legend()
        fig.savefig(f'flatfield_correction_{win}.png')
    
    return da