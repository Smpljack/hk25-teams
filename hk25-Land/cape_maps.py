#%%
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import dask.array as da
import numcodecs
import healpix as hp
import easygems.healpix as egh
import easygems.remap as egr
import intake     # For catalogs


def worldmap(var, **kwargs):
    #projection = ccrs.Robinson(central_longitude=-135.5808361)
    projection = ccrs.Robinson(central_longitude=0)
    fig, ax = plt.subplots(
        figsize=(8, 4), subplot_kw={"projection": projection}, 
        constrained_layout=True
    )
    ax.set_global()

    hpshow = egh.healpix_show(var, ax=ax, **kwargs)
    cbar = plt.colorbar(hpshow, ax=ax, orientation='vertical', 
                    pad=0.05, shrink=0.8)
    ax.coastlines()
    return fig, ax

def get_axis_europe():
    """Create cartyop.GeoAxis with an extent centered around Europe."""
    _, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    ax.coastlines()
    ax.set_extent([-40, 60, 20, 65])

    return ax

def plot_extent_box(ax, extent, edgecolor="m", linewidth=2):
    """Add a rectangular patch around an extent to a cartopy.GeoAxis."""
    return ax.add_patch(
        plt.Rectangle(
            xy=[extent[0], extent[2]],
            width=extent[1] - extent[0],
            height=extent[3] - extent[2],
            facecolor="none",
            edgecolor=edgecolor,
            linewidth=linewidth,
            transform=ccrs.PlateCarree(),
        )
    )
def get_healpix_region(extent, zoom):
# Compute HEALPix lat/lon coordinates for a given order (zoom level)
    order = zoom
    nside = hp.order2nside(order)
    npix = hp.nside2npix(nside)

    hp_lon, hp_lat = hp.pix2ang(
        nside, np.arange(npix), nest=True, lonlat=True)
    hp_lon = (hp_lon + 180) % 360 - 180

    # Find all grid points within the defined extent
    icell, = np.where(
        (hp_lon > extent[0]) &
        (hp_lon < extent[1]) &
        (hp_lat > extent[2]) &
        (hp_lat < extent[3])
    )

    region = np.full_like(hp_lon, fill_value=np.nan)  # Full HEALPix with NaNs
    region[icell] = 1  # Set selected values to 1
    return icell, region

def load_native_xsh24_data(
        zoom=7,
        cat_name='/home/tmerlis/hackathon/xsh24_scream_main.yaml'):
    combo_cat = intake.open_catalog(cat_name)
    dsn = combo_cat.xsh24_native(zoom=zoom).to_dask()
    dsn = dsn.pipe(egh.attach_coords)
    return dsn

def load_coarse_xsh24_data(
        zoom=7,
        cat_name='/home/tmerlis/hackathon/xsh24_scream_main.yaml'):
    combo_cat = intake.open_catalog(cat_name)
    ds = combo_cat.xsh24_coarse(zoom=zoom).to_dask()
    ds = ds.pipe(egh.attach_coords)
    return ds

def load_coarse_scream_data(
        zoom=7,
        cat_name='/home/tmerlis/hackathon/xsh24_scream_main.yaml'):
    combo_cat = intake.open_catalog(cat_name)
    ds = combo_cat.scream_ne120(zoom=zoom).to_dask()
    ds = ds.pipe(egh.attach_coords)
    return ds

def get_quantile(da, q):
    return xr.apply_ufunc(
        lambda da, q: np.quantile(da, q, axis=1), da, q,
                                  input_core_dims =[['cell', 'dayofyear'], []], 
                                  output_core_dims=[['cell']])

def find_regional_cape_maxima(regional_cape_da, n_cases=10):
    """Find the n_cases with the highest cape in the regional_cape_da."""
    return regional_cape_da.sortby(regional_cape_da, ascending=False).isel(case=slice(0, n_cases))

#%%
if __name__ == "__main__":
    zoom = 5
    ds = load_native_xsh24_data(zoom=zoom).sel(time='2020')
    daily_max_cape = ds.CAPE_max.groupby('time.dayofyear').max()
    daily_max_cape_97 = get_quantile(daily_max_cape.load(), 0.97)
    #%%
    # Select region
    extent = [-20, 40, 30, 60]  # Europe
    icell, region = get_healpix_region(extent, zoom)
    ds_region = ds.isel(cell=icell)
    regional_cape_maxima = find_regional_cape_maxima(
        ds_region.CAPE_max.stack(case=['cell', 'time']))
    #%%
    fig, ax = worldmap(daily_max_cape_97)
    ax.scatter(regional_cape_maxima.lon.values, regional_cape_maxima.lat.values, 
               c=regional_cape_maxima.values, cmap='RdYlBu_r')
    # %%