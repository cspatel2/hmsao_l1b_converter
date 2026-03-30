#%%
from calendar import c
from curses import window
from os import error
from unittest import skip

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# %%
method = ''

#center of mass method for line profile
if method == 'COM':
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

    
    fns = list(Path('~/locsststor/proc/hmsao-v1/l1a').expanduser().glob('**/*.nc'))
    fns.sort()
    win = '5577'
    
    ds = xr.open_dataset(fns[0])
    # 
    #find boundaries
    ids = ds.countrate.sum('tstamp').clip(min=0)
    ids.data = gaussian_filter(ids.data, sigma = 5)

    vmin = np.nanpercentile(ids, .1)
    vmax = np.nanpercentile(ids, 99.99)
    #
    ids.plot(vmin = vmin, vmax = vmax)
    width = 0.2
    xmin = int(win)/10 - width/2
    xmax = xmin + width
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
    #
    tds = ids.sel(wavelength = slice(bounds['line'][0], bounds['line'][1]))
    bgds = ids.sel(wavelength = slice(bounds['bg'][0], bounds['bg'][1]))
    idx = np.min([tds.wavelength.shape[0], bgds.wavelength.shape[0]])
    tds = tds.isel(wavelength = slice(0, idx))
    bgds = bgds.isel(wavelength = slice(0, idx))
    tds.data = tds.data - bgds.data
    tds = tds.clip(min=0)
    #
    norm = xr.apply_ufunc(
        center_of_mass_position,
        tds.wavelength,
        tds,
        input_core_dims=[['wavelength'], ['wavelength']],
        vectorize=True,
    )
    #
    norm = norm.sel(za = slice(-17,15))
    norm -= (norm.max())
    # norm /= (norm.min())
    norm = norm.to_dataset(name = 'line_profile')
    norm.line_profile.plot(y = 'za')

    #
    savefn = 'line_profile_{win}.nc'
    norm.to_netcdf(savefn.format(win= win))

    #
    testds = xr.open_dataset('line_profile_5577.nc')
    #
    testds

    testds.line_profile.plot(y = 'za')
# %%
#find peaks method
fns = list(Path('~/locsststor/proc/hmsao-v1/l1a').expanduser().glob('**/*.nc'))
fns.sort()
win = '5577'
ds = xr.open_dataset(fns[0])
da = ds.countrate.sum('tstamp', skipna = True).clip(min=0)

# %%
ida = da.copy()
ida.data = gaussian_filter(ida.data, sigma = 5)
# %%
ida.plot()



# %%
def gaussian(x, a,xo, sigma,r):
    return a* np.exp(-(x-xo)**2/(2*sigma**2)) + r

#%%
zaslice = slice(-17,15)
# wlslice = slice(int(win)/10 - 0.1, int(win)/10 + 0.1)
ida = ida.sel(za = zaslice)
da = da.sel(za = zaslice)
ida.plot()
plt.figure()
da.plot()

#%%
fitted = ida.curvefit(coords='wavelength', func=gaussian, skipna=True,
                      p0={'a': da.max(skipna=True) - da.min(skipna=True), 
                          'xo': int(win)/10, 
                          'sigma': .3}, errors='ignore')
# %%
fitted.curvefit_coefficients.sel(param = 'xo').plot(y = 'za')
# %%
zidx = -500
popt = fitted.curvefit_coefficients.isel(za = zidx)

da.isel(za = zidx).plot()
x = da.wavelength.values
plt.scatter(x, gaussian(x, *popt.values), s = .5, color = 'r')
# %%
fitted.curvefit_coefficients.sel(param = 'xo').plot(y = 'za')
# %%
lp = fitted.curvefit_coefficients.sel(param = 'xo')
# %%
glp = lp.copy()
glp.data = gaussian_filter(lp, sigma = 20)
# %%
lp.plot(y = 'za')   
glp.plot(y = 'za')
# %%
fitted.curvefit_coefficients.sel(param = 'xo').plot(y = 'za')



#################################################################################################
#################################################################################################

#################################################################################################
#line profile from obsorption line
# %%
win = '5577'
zaidx = -500
zaslice = slice(-17,15)
wlslice = slice(int(win)/10 - 0.1, int(win)/10 + 0.06)
# wlslice = slice(int(win)/10 - 0.1, int(win)/10 + 0.1)
#%%
#########################################################################################
p = Path('~/locsststor/proc/hmsao-v1/l1a').expanduser()
win = '6563'
fns = list(p.glob(f'**/*{win}*.nc'))
fns.sort()
ds = xr.open_dataset(fns[0])
# wlslice = slice(ds.wavelength.min()+1.1, ds.wavelength.max()-1)
da = ds.countrate.sel(wavelength = wlslice, za = zaslice).sum('tstamp', skipna = True).clip(min=0)
#%%
da
#%%
nda = da.copy()
nda -= (nda[-1] - nda[0])
nda /= (nda.max(skipna=True) - nda.min(skipna=True))
#%%
q = da.quantile(0.5, dim="wavelength")
mask = da >= q

masked = da.where(mask)

poly = masked.polyfit(dim="wavelength", deg=1, skipna=True)
cont = xr.polyval(da["wavelength"], poly.polyfit_coefficients)
#%%
da.isel(za = zaidx).plot()
cont.isel(za = zaidx).plot()
#%%
norm = 1 - da/cont
#%%
norm.isel(za = zaidx).plot()
#%%
# pfit = nda.polyfit(dim='wavelength', deg = 2, skipna=True)
# cont = xr.polyval(nda["wavelength"], pfit.polyfit_coefficients)


#%%
# nda.isel(za = zaidx).plot()
# cont.isel(za = zaidx).plot()

# #%%
# (nda - cont).isel(za = zaidx).plot()


#%%
# continuum = (
#     nda.rolling(wavelength=25)
#       .construct("window")
#       .quantile(0.95, dim="window")
# )
#%%
# continuum.isel(za = zaidx).plot()
# #%%
# norm = 1 - nda/continuum
# norm.isel(za = zaidx).plot()
#%%
def Gaussian(x, a,xo, sigma,r):
    return a* np.exp(-(x-xo)**2/(2*sigma**2)) + r

fitted = norm.curvefit(coords='wavelength', func=Gaussian, skipna=True,
                      p0={'a': np.abs(np.abs(norm.max(skipna=True)) - np.abs(norm.min(skipna=True))), 
                          'xo': int(win)/10, 
                          'sigma': .1,
                          'r': 0}, errors='ignore')
#%%
popt = fitted.curvefit_coefficients.isel(za = zaidx)
norm.isel(za = zaidx).plot()
x = norm.wavelength.values
plt.scatter(x, Gaussian(x, *popt.values), s = .5, color = 'r')
#%%
xo = fitted.curvefit_coefficients.sel(param = 'xo')
# xo = xo.clip(np.nanpercentile(xo, 2), np.nanpercentile(xo, 80))

#%%
xs = xo.copy()
xs.data = gaussian_filter1d(xo.data, sigma = 10)
# xs = xs.interp(za = np.arange(xs.za.min(), xs.za.max(), 0.1), method = 'linear')
xo.plot(y = 'za')
xs.plot(y = 'za')   

#%%























#%%
#%
#%%
poly = da.polyfit(dim='wavelength', deg = 1, skipna=True)
#%%
da.isel(za = zaidx).plot()
x = da.wavelength.values
y = np.polyval(poly.polyfit_coefficients.isel(za = zaidx).values,x)
plt.plot(x, y, color = 'r')

continuum = xr.polyval(da["wavelength"], poly.polyfit_coefficients)




# %%
da.isel(za = zaidx).plot()
continuum.isel(za = zaidx).plot()
# %%
subds = da.copy()
subds.data = da - continuum
norm = 1 - subds/(subds.max(skipna=True) - subds.min(skipna=True))
# norm = 1 - da/continuum
# %%

#%%
wlslice = slice(None,557.72)
norm = norm.sel(wavelength = wlslice)

cont = norm.polyfit(dim='wavelength', deg = 1, skipna=True)
norm.data = norm - xr.polyval(norm["wavelength"], cont.polyfit_coefficients)
norm.isel(za = zaidx).plot()

# %%
def Gaussian(x, a,xo, sigma,r):
    return a* np.exp(-(x-xo)**2/(2*sigma**2)) + r

fitted = norm.curvefit(coords='wavelength', func=Gaussian, skipna=True,
                      p0={'a': np.abs(np.abs(norm.max(skipna=True)) - np.abs(norm.min(skipna=True))), 
                          'xo': int(win)/10, 
                          'sigma': .1,
                          'r': 0}, errors='ignore')
# %%
zaidx = 100
norm.isel(za = zaidx).plot()
x = norm.wavelength.values
popt = fitted.curvefit_coefficients.isel(za = zaidx)
plt.scatter(x, Gaussian(x, *popt.values), s = .5, color = 'r')
# %%
fitted.curvefit_coefficients.sel(param = 'xo').plot(y = 'za')
# %%
xo = fitted.curvefit_coefficients.sel(param = 'xo')
xo = xo.clip(np.nanpercentile(xo, 2), np.nanpercentile(xo, 99))
xs = xo.copy()
xs.data = gaussian_filter1d(xo.data, sigma = 100)
xo.plot(y = 'za')
xs.plot(y = 'za')
plt.x

# %%
da.plot()
# %%
ds = xr.open_dataset(fns[3+7])
wlmin = np.max([ds.wavelength.min(), int(win)/10 - 0.1])
wlmax = np.min([ds.wavelength.max(), int(win)/10 + .9])
wlslice = slice(wlmin, wlmax)
da = ds.countrate.sel(wavelength = wlslice, za = zaslice).sum('tstamp', skipna = True).clip(min=0)
# %%
da.isel(za = zaidx).plot()
# %%
q = da.quantile(0.4, dim="wavelength")
mask = da >= q
masked = da.where(mask)
poly = masked.polyfit(dim="wavelength", deg=1, skipna=True)
cont = xr.polyval(da["wavelength"], poly.polyfit_coefficients)
# %%
da.isel(za = zaidx).plot()
cont.isel(za = zaidx).plot()
# %%
nda = da.copy()
nda /=  cont
nda = 1 - nda
# %%
nda.isel(za = zaidx).plot()

# %%
cds = nda.coarsen(za = 20, boundary = 'trim').sum()
# %%
signal = cds.isel(za = 10)
peaks,props = find_peaks(signal, height = 0.5, distance = 5)
# %%
signal.plot()
plt.scatter(signal.wavelength.values[peaks], signal.values[peaks], color = 'r')
# %%
def rsquared_for_gaussian(x, y, popt):
    residuals = y - Gaussian(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

wlidx = peaks[2]
windowsize = 0.1
testda = cds.sel(wavelength = slice(signal.wavelength.values[wlidx] - windowsize, signal.wavelength.values[wlidx] + windowsize))
fitda = testda.curvefit(coords='wavelength', func=Gaussian, skipna=True,
                      p0={'a': np.abs(np.abs(testda.max(skipna=True)) - np.abs(testda.min(skipna=True))), 
                          'xo': signal.wavelength.values[wlidx], 
                          'sigma': windowsize/2,
                          'r': 0}, errors='ignore')

# %%
calcda = Gaussian(testda.wavelength.values, *fitda.curvefit_coefficients.isel(za = 10).values)
# %%
testda.isel(za = 10).plot()
plt.scatter(testda.wavelength.values, calcda, color = 'r', s = .5)
# %%
scores = []
for p in peaks:
    testda = cds.sel(wavelength = slice(signal.wavelength.values[p] - windowsize, signal.wavelength.values[p] + windowsize))
    fitda = testda.curvefit(coords='wavelength', func=Gaussian, skipna=True,
                        p0={'a': np.abs(np.abs(testda.max(skipna=True)) - np.abs(testda.min(skipna=True))), 
                            'xo': signal.wavelength.values[p], 
                            'sigma': windowsize/2,
                            'r': 0}, errors='ignore')
    score = rsquared_for_gaussian(testda.wavelength.values, testda.isel(za = 10).values, fitda.curvefit_coefficients.isel(za = 10).values)
    scores.append(score)
# %%
winner = nda.wavelength.values[peaks[np.argmax(scores)]]
winner_slice = slice(winner - windowsize, winner + windowsize)
# %%
nda.sel(wavelength = winner_slice).isel(za = 70).plot()
plt.axvline(winner, color = 'g', ls = '--')

# %%
#########################################################
win = '6563'
p = Path('~/locsststor/proc/hmsao-v1/l1a').expanduser()
fns = list(p.glob(f'**/*{win}*.nc'))
fns.sort()
ds = xr.open_dataset(fns[3+7])
# %%
wlslice = slice(ds.wavelength.min()+1.1, ds.wavelength.max()-.9)
da =ds.countrate.sel(za = slice(-17,15), wavelength = wlslice).isel(tstamp = 50)
# %%
da.plot()
# 
# %%
da.sum('za').plot()
# %%
## remove continuum
q = da.sum('za').quantile(0.4, dim="wavelength")
mask = da.sum('za') >= q
masked = da.sum('za').where(mask)
poly = masked.polyfit(dim="wavelength", deg=2, skipna=True)
cont = xr.polyval(da["wavelength"], poly.polyfit_coefficients)
#%%
norm = 1 - da/cont
# %%
da.sum('za').plot()
cont.plot()
plt.figure()
norm.mean('za').plot()
# %%
signal = norm.mean('za')
peaks,_ = find_peaks(signal, height = 0.9, distance = 100)
# signal.plot()
# plt.scatter(norm.wavelength.values[peaks], norm.mean('za').values[peaks], color = 'r')
# %%
def Gaussian(x, a,xo, sigma,r):
    return a* np.exp(-(x-xo)**2/(2*sigma**2)) + r




scores = []
windowsize = 0.3
for p in peaks:
    testda = norm.sel(wavelength = slice(signal.wavelength.values[p] - windowsize, signal.wavelength.values[p] + windowsize))
    fitda = testda.curvefit(coords='wavelength', func=Gaussian, skipna=True,
                        p0={'a': np.abs(np.abs(testda.max(skipna=True)) - np.abs(testda.min(skipna=True))), 
                            'xo': signal.wavelength.values[p], 
                            'sigma': windowsize/2,
                            'r': 0}, errors='ignore')
    fitda = fitda.mean('za') #get the mean curve fit across all zaslices
    # plt.figure()
    # testda.mean('za').plot()
    # calcda = Gaussian(testda.wavelength.values, *fitda.curvefit_coefficients.values)
    plt.scatter(testda.wavelength.values, calcda, color = 'r', s = .5)
    score = rsquared_for_gaussian(testda.wavelength.values, testda.values, fitda.curvefit_coefficients.values)
    scores.append(score)

# %%
central_wl = da.wavelength.values[peaks[np.argmax(scores)]]

# %%
windowsize = 0.25
wlslice = slice(central_wl - windowsize, central_wl + windowsize)
norm = norm.sel(wavelength = wlslice)
fitted = norm.curvefit(coords='wavelength', func=Gaussian, skipna=True,
                      p0={'a': np.abs(np.abs(norm.max(skipna=True)) - np.abs(norm.min(skipna=True))), 
                          'xo': central_wl, 
                          'sigma': windowsize/2,
                          'r': 0}, errors='ignore')
# %%
da.plot()
fitted.curvefit_coefficients.sel(param = 'xo').plot(y = 'za')
# %%
norm.isel(za = 10).plot()
x = norm.wavelength.values
popt = fitted.curvefit_coefficients.isel(za = 10)
plt.scatter(x, Gaussian(x, *popt.values), s = .5, color = 'r')
# %%
