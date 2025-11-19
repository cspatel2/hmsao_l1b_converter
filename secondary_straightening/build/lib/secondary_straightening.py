#taken in L1a files and performs secondary straightening of the spectra. Might truncate the spectra for the zenith angle range the line profile is defined for.
#%%
from curses import window
import xarray as xr
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from skimage import transform
from matplotlib import pyplot as plt

#%%
def secondary_straightening_core(img:np.ndarray, line_profile:np.ndarray, wlppix:float) -> np.ndarray:
    """ performs a skimage.tranform.warp() on an image to straighten the spectra.

    Args:
        img (np.ndarray): image of shape (za, wavelength)
        line_profile (np.ndarray): line profile of shape (za,)
        wlppix (float): wavelength per pixel (nm/pixel) 

    Returns:
        np.ndarray: straightened image of same shape as input img
    """    
    #create meshgrid of pixel indices for output image coordinates
    xpix = np.arange(img.shape[0])
    ypix = np.arange(img.shape[1])
    mxo, myo = np.meshgrid(xpix, ypix, indexing='ij')

    #create a modified y-coordinate map based on the line profile as the inverse of the distortion in the input image
    lp_pix = line_profile/ wlppix
    mi = myo + lp_pix[np.newaxis,:].T

    #create the inverse coordinate traform map (output coords -> input coords)
    imap = np.zeros((2,*(img.shape)))
    imap[0,:,:] = mxo #output x-coords map remains the same
    imap[1,:,:] = mi #modified output y-coords map

    straightened = transform.warp(img, imap, order=1, mode='edge', cval=np.nan)
    return straightened

#%%
def secondary_straightening(ds:xr.Dataset, lprof:xr.Dataset) -> xr.Dataset:
    """ performs secondary straightening on ds of dims (...,za,wavelength) using skiimage.transform.warp() in the core function.

    Args:
        ds (xr.Dataset): dataset containing the image with dims ( .., za, wavelength)
        lprof (xr.Dataset): dataset containing the line profile with dim (za,)

    Returns:
        xr.Dataset: dataset containing the straightened image with same dims as input ds
    """    
    #select only the za range for which the line profile is defined
    ds = ds.sel(za = lprof.za)
    #nm per pixel to convert wavelength axis back to pixel units
    wlppix = float(np.mean(np.diff(ds.wavelength.data)))

    id = 'countrate'

    straightened_data = xr.apply_ufunc(
        secondary_straightening_core,
        ds[id],
        lprof.line_profile,
        input_core_dims=[['za', 'wavelength'], ['za']],
        output_core_dims=[['za', 'wavelength']],
        kwargs={'wlppix': wlppix},
        vectorize=True,
    )
    ds[id].data = straightened_data.data
    return ds
# %%
EXAMPLE = False
if EXAMPLE:
    win = '5577'
    # wlslice = slice(629.5, 630.5)
    wlslice = slice(557.3, 557.9)
    PLOT = True
    filepath =  Path(f'/home/sunip/Codes/charmi/hms-ao/l1a_converter/data/l1a/202503/hmsao_l1a_20250320_{win}[0].nc')
    ds = xr.open_dataset(filepath)
    lp = xr.open_dataset(Path(f'line_profile_{win}.nc'))
    ss = secondary_straightening(ds.isel(tstamp = slice(0,5)), lp)
    if PLOT:
        fig,ax = plt.subplots(1,2,figsize=(12,6))
        ds.intensity.isel(tstamp=0).sel(wavelength=wlslice, za=slice(-17,15)).plot(ax=ax[0], y = 'za', vmin = 0)
        ax[0].set_title('Original')
        ss.intensity.isel(tstamp=0).sel(wavelength=wlslice).plot(ax=ax[1], y = 'za', vmin = 0)
        ax[1].set_title('Straightened')
        plt.show()
    ss.tstamp.attrs = ds.tstamp.attrs
    ss.wavelength.attrs = ds.wavelength.attrs
    ss.za.attrs = ds.za.attrs
##%%
# %%
