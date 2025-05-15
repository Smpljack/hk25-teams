#%%
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import dask.array as da
import numcodecs
import healpix as hp
import healpy as hpy
from healpy.visufunc import projscatter
import easygems.healpix as egh
import easygems.remap as egr
import intake     # For catalogs


def worldmap(var, **kwargs):
    #projection = ccrs.Robinson(central_longitude=-135.5808361)
    projection = ccrs.Robinson(central_longitude=0)
    fig, ax = plt.subplots(
        figsize=(8, 4), 
        constrained_layout=True,
        subplot_kw={"projection": projection}
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
    # Group by time and get max cape for each day
    daily_max = regional_cape_da.resample(time='1D').max().max('cell')
    sorted_idx = np.argsort(daily_max.values)[-n_cases:][::-1]
    top_times = daily_max['time'].values[sorted_idx]
    top_events = []
    for day in top_times:
        day_data = regional_cape_da.sel(time=day)
        idx = day_data.argmax('cell').compute().item()
        top_events.append({
            'lon': day_data.lon.isel(cell=idx).compute().item(),
            'lat': day_data.lat.isel(cell=idx).compute().item(),
            'cape': day_data.isel(cell=idx).compute().item(),
            'date': str(day)[:10]
        })
    events_da = xr.DataArray(
        data=np.array([e['cape'] for e in top_events]),
        coords={
            'lon': ('event', [e['lon'] for e in top_events]),
            'lat': ('event', [e['lat'] for e in top_events]),
            'date': ('event', [e['date'] for e in top_events])
        },
        dims=['event']
    )
    return events_da


def find_regional_cape_maxima2(regional_cape_da, n_cases=10):
    """
        Alternative method to calculate regional cape maxima (which saves
        info about.

        Note: I've been running this with regional_cape_da loaded into memory.
    """
    # For each dayofyear, get location where CAPE maximizes in the region
    icell_locs = regional_cape_da.argmax('cell')

    # Sort icell_locs by the value of CAPE on that day, save top n_case events
    events_da = icell_locs.sortby(regional_cape_da.isel(cell=icell_locs),
                                  ascending=False).isel(dayofyear=np.arange(n_cases))
    
    return events_da

def boxes_around_events(lons, lats, box_km=10):
    """
    Create a list of [min_lon, max_lon, min_lat, max_lat] boxes around each (lon, lat) event.
    Each box is box_km x box_km in size (default 10 km).
    """

    # Earth's radius in km
    R = 6371.0
    half_side = box_km / 2.0
    
    boxes = []
    for lon, lat in zip(lons, lats):
        # Latitude: 1 deg ≈ 111.32 km
        dlat = (half_side / 111.32)
        # Longitude: 1 deg ≈ 111.32 * cos(latitude) km
        dlon = half_side / (111.32 * np.cos(np.radians(lat)))
        min_lat = lat - dlat
        max_lat = lat + dlat
        min_lon = lon - dlon
        max_lon = lon + dlon
        boxes.append([min_lon, max_lon, min_lat, max_lat])
    return boxes


if __name__ == "__main__":
    zoom = 8
    ds = load_native_xsh24_data(zoom=zoom).sel(time=slice('2020-01-01', '2020-12-31'))
    daily_max_cape = ds.CAPE_max.groupby('time.dayofyear').max('time')
    daily_max_cape_97 = get_quantile(daily_max_cape.load(), 0.97)
    #%%
    # Select region
    extent_sa = [35, 60, 10, 35]  # Saudi Arabia
    extent_na = [-110, -70, 30, 55]
    extent_au = [140, 170, -50, -25]
    icell, region = get_healpix_region(extent_sa, zoom)
    ds_sa = ds.isel(cell=icell)
    icell, region = get_healpix_region(extent_na, zoom)
    ds_na = ds.isel(cell=icell)
    icell, region = get_healpix_region(extent_au, zoom)
    ds_au = ds.isel(cell=icell)
    sa_cape_maxima = find_regional_cape_maxima(
        ds_sa.CAPE_max,
        n_cases=20)
    na_cape_maxima = find_regional_cape_maxima(
        ds_na.CAPE_max,
        n_cases=20)
    au_cape_maxima = find_regional_cape_maxima(
        ds_au.CAPE_max,
        n_cases=20)
    #%%
    fig, axs = plt.subplots(nrows=2, ncols=2,
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(20, 15))
    scatter_cmap = 'YlGnBu'
    axs[0, 0].set_global()
    c = egh.healpix_show(daily_max_cape_97, ax=axs[0, 0], cmap='Reds')
    plot_extent_box(axs[0, 0], extent_sa, edgecolor="g", linewidth=2)
    axs[0, 0].scatter(
        sa_cape_maxima.lon, sa_cape_maxima.lat, 
        c=sa_cape_maxima, cmap=scatter_cmap, s=10, 
        vmin=0, vmax=7000, alpha=0.5)
    plot_extent_box(axs[0, 0], extent_na, edgecolor="c", linewidth=2)
    axs[0, 0].scatter(
        na_cape_maxima.lon, na_cape_maxima.lat, 
        c=na_cape_maxima, cmap=scatter_cmap, s=10, 
        vmin=0, vmax=7000, alpha=0.5)
    plot_extent_box(axs[0, 0], extent_au, edgecolor="m", linewidth=2)
    s = axs[0, 0].scatter(
        au_cape_maxima.lon, au_cape_maxima.lat, 
        c=au_cape_maxima, cmap=scatter_cmap, s=10, 
        vmin=0, vmax=7000, alpha=0.5)
    
    axs[0, 1].set_extent(extent_na)
    egh.healpix_show(daily_max_cape_97, ax=axs[0, 1], cmap='Reds')
    axs[0, 1].scatter(
        na_cape_maxima.lon, na_cape_maxima.lat, 
        c=na_cape_maxima, cmap=scatter_cmap, s=100, 
        vmin=0, vmax=7000, alpha=0.5)
    
    axs[1, 0].set_extent(extent_sa)
    egh.healpix_show(daily_max_cape_97, ax=axs[1, 0], cmap='Reds')
    axs[1, 0].scatter(
        sa_cape_maxima.lon, sa_cape_maxima.lat, 
        c=sa_cape_maxima, cmap=scatter_cmap, s=100, 
        vmin=0, vmax=7000, alpha=0.5)

    axs[1, 1].set_extent(extent_au)
    egh.healpix_show(daily_max_cape_97, ax=axs[1, 1], cmap='Reds')
    axs[1, 1].scatter(
        au_cape_maxima.lon, au_cape_maxima.lat, 
        c=au_cape_maxima, cmap=scatter_cmap, s=100, 
        vmin=0, vmax=7000, alpha=0.5)
    
    # Create two axes for colorbars on the right side
    cbar_ax1 = fig.add_axes([0.92, 0.55, 0.02, 0.3])  # For first colorbar
    cbar_ax2 = fig.add_axes([0.92, 0.15, 0.02, 0.3])  # For second colorbar
    
    # Add colorbars to the new axes
    plt.colorbar(c, cax=cbar_ax1, orientation='vertical')
    plt.colorbar(s, cax=cbar_ax2, orientation='vertical')
    cbar_ax1.tick_params(labelsize=12)
    cbar_ax2.tick_params(labelsize=12)
    cbar_ax1.set_ylabel('P97 of daily max. CAPE [J/kg]', fontsize=12)
    cbar_ax2.set_ylabel('peak CAPE [J/kg]', fontsize=12)
    for ax, color in zip(axs.flatten(), ['k', 'c', 'g', 'm']):
        ax.coastlines()
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(6)
    # plt.tight_layout()
    # plt.colorbar(s, ax=ax, orientation='vertical', 
    #              pad=0.01, shrink=0.3, 
    #              label='max. daily CAPE [J/kg]')

    # %%
