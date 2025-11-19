#%%
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from pathlib import Path

# %%

# put in an array of counts vs wavelength
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
fn = Path('/home/sunip/Codes/charmi/hms-ao/l1a_converter/data/l1a/202503/hmsao_l1a_20250320_7774[0].nc')
win = fn.stem.split('_')[-1].strip('[0]')
# %%
ds = xr.open_dataset(fn)
# %%
#find boundaries
ids = ds.intensity.sum('tstamp').clip(min=0)
vmin = np.nanpercentile(ids, .1)
vmax = np.nanpercentile(ids, 99.99)
#%%
ids.plot(vmin = vmin, vmax = vmax)
xmin = 777.4
xmax = 777.74
plt.axvline(xmin, color = 'w', ls = '--', lw = 0.3)
plt.axvline(xmax, color = 'w', ls = '--', lw = 0.3)

xbgmin = xmax + 0.05
xbgmax = xbgmin + (xmax - xmin)
plt.axvline(xbgmin, color = 'w', ls = '--', lw = 0.3)
plt.axvline(xbgmax, color = 'w', ls = '--', lw = 0.3)

bounds = {
    'line': (xmin, xmax),
    'bg': (xbgmin, xbgmax)
}
print(bounds)
# %%
tds = ids.sel(wavelength = slice(bounds['line'][0], bounds['line'][1]))
bgds = ids.sel(wavelength = slice(bounds['bg'][0], bounds['bg'][1]))
idx = np.min([tds.wavelength.shape[0], bgds.wavelength.shape[0]])
tds = tds.isel(wavelength = slice(0, idx))
bgds = bgds.isel(wavelength = slice(0, idx))
tds.data = tds.data - bgds.data
tds = tds.clip(min=0)
# %%
norm = xr.apply_ufunc(
    center_of_mass_position,
    tds.wavelength,
    tds,
    input_core_dims=[['wavelength'], ['wavelength']],
    vectorize=True,
)
#%%
norm = norm.sel(za = slice(-17,15))
norm -= (norm.max())
# norm /= (norm.min())
norm = norm.to_dataset(name = 'line_profile')
norm.line_profile.plot(y = 'za')

# %%
savefn = 'line_profile_{win}.nc'
norm.to_netcdf(savefn.format(win= win))

# %%
testds = xr.open_dataset('line_profile_5577.nc')
# %%
testds
# %%
testds.line_profile.plot(y = 'za')
# %%
