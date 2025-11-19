#%%
import numpy as np
import xarray as xr
from misdesigner import MisInstrumentModel, MisCurveRemover
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from skimage import transform
# %%
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
#%%
filepath =  Path('/home/sunip/Codes/charmi/hms-ao/l1a_converter/data/l1a/202503/hmsao_l1a_20250320_5577[0].nc')
profilepath = Path('line_profile_5577.nc')
modelpath = '../hmsa_origin_ship.json'
# %%
# %%
win = filepath.stem.split('_')[-1].strip('[0]')
window = float(int(win)/10)
# %%
# Create model and confirm that the Instrument file provided works
# model = MisInstrumentModel.load(modelpath)
# predictor = MisCurveRemover(model)  # line straightening
# windows = predictor.windows
# %%
lprof = xr.open_dataset(profilepath)
ds = xr.open_dataset(filepath)
ds = ds.sel(za = lprof.za)

lprof.line_profile.plot(y = 'za')
plt.axvline(0, color = 'r', ls = '--')
# %%
lprof.line_profile.data = gaussian_filter1d(lprof.line_profile.data, sigma=10, axis=0)
# %%
lprof.line_profile.plot(y = 'za')
plt.axvline(0, color = 'r', ls = '--')

# %%
#%%
img = ds.intensity.isel(tstamp = 0).copy()
xpix = np.arange(ds.za.shape[0])
ypix = np.arange(ds.wavelength.shape[0])
mxi, myi = np.meshgrid(xpix, ypix, indexing='ij')
#%%

lp_pix = lprof.line_profile.data/ np.mean(np.diff(ds.wavelength.data)) 
mo = myi + lp_pix[np.newaxis, :].T

# %%
imap = np.zeros((2,*(img.shape)))
imap[0,:,:] = mxi
imap[1,:,:] = mo

wrapped = transform.warp(
    img.data,
    inverse_map = imap,
    order = 1,
    cval = np.nan,
)
# %%
fig,ax = plt.subplots(1,2, figsize = (10,5))
im0 = ax[0].imshow(img.data, vmin = 0, vmax = np.nanpercentile(img.data, 99.99))
fig.colorbar(im0, ax=ax[0])
im1 = ax[1].imshow(wrapped, vmin = 0, vmax = np.nanpercentile(wrapped, 99.99))
fig.colorbar(im1, ax=ax[1])
# %%

wds = img.copy()
wds.data = wrapped
# %%
wds.plot(vmin = 0)
plt.axvline(window, color = 'w', ls = '--')

# %%
img = img.sel(wavelength = slice(window - 0.1, window + 0.1))
wds = wds.sel(wavelength = slice(window - 0.1, window + 0.1))
com_before = xr.apply_ufunc(
    center_of_mass_position,
    img.wavelength,
    img,
    input_core_dims=[['wavelength'], ['wavelength']],
    vectorize=True,
)

com_after = xr.apply_ufunc(
    center_of_mass_position,
    wds.wavelength,
    wds,
    input_core_dims=[['wavelength'], ['wavelength']],
    vectorize=True,
)
com_before.data -= com_before.max().values
com_after.data -= com_after.max().values
# %%

com_before.plot(y = 'za')
com_after.plot(y = 'za')


# %%
img.plot(vmin = 0, vmax = np.nanpercentile(img.data, 99.99))
# %%
wds.plot(vmin = 0, vmax = np.nanpercentile(wds.data, 99.99))  
# %%
