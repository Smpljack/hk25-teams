#%%
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import healpix as hp
import easygems.healpix as egh
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

def find_regional_cape_maxima(regional_cape_da, n_cases=20):
    """
    Find the top n_cases CAPE maxima on unique days, preserving the full timestamp.
    Much more efficient for dask/xarray.
    """
    # Load data into memory (if not already)
    regional_cape_da = regional_cape_da.load()
    # Flatten over cell and time
    flat = regional_cape_da.stack(event=('cell', 'time'))
    cape_vals = flat.values
    cell_vals = flat['cell'].values
    time_vals = flat['time'].values

    # Sort all events by CAPE descending
    sorted_idx = np.argsort(cape_vals)[::-1]
    selected_dates = set()
    top_events = []
    for idx in sorted_idx:
        time = time_vals[idx]
        date_str = str(time)[:10]
        if date_str in selected_dates:
            continue
        selected_dates.add(date_str)
        cell = cell_vals[idx]
        lon = regional_cape_da['lon'].sel(cell=cell).item()
        lat = regional_cape_da['lat'].sel(cell=cell).item()
        cape = cape_vals[idx]
        top_events.append({
            'lon': lon,
            'lat': lat,
            'cape': cape,
            'date': date_str,
            'timestamp': str(time)
        })
        if len(top_events) >= n_cases:
            break
    # Convert top_events list of dicts to a DataArray
    event_data = xr.DataArray(
        data=[e['cape'] for e in top_events],
        coords={
            'lat': ('event', [e['lat'] for e in top_events]),
            'lon': ('event', [e['lon'] for e in top_events]),
            'time': ('event', [np.datetime64(e['timestamp']) for e in top_events])
        },
        dims=['event']
    )
    return event_data


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



#%%
def maxima_to_arrays(maxima):
    """Convert list of maxima dicts to arrays for plotting."""
    lons = [e['lon'] for e in maxima]
    lats = [e['lat'] for e in maxima]
    capes = [e['cape'] for e in maxima]
    return lons, lats, capes
#%%
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
    # Convert maxima lists to arrays for plotting
    sa_lons, sa_lats, sa_capes = maxima_to_arrays(sa_cape_maxima)
    na_lons, na_lats, na_capes = maxima_to_arrays(na_cape_maxima)
    au_lons, au_lats, au_capes = maxima_to_arrays(au_cape_maxima)

    #%%
    fig, axs = plt.subplots(nrows=2, ncols=2,
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(20, 15))
    scatter_cmap = 'YlGnBu'
    axs[0, 0].set_global()
    c = egh.healpix_show(daily_max_cape_97, ax=axs[0, 0], cmap='Reds')
    plot_extent_box(axs[0, 0], extent_sa, edgecolor="g", linewidth=2)
    axs[0, 0].scatter(
        sa_lons, sa_lats, 
        c=sa_capes, cmap=scatter_cmap, s=10, 
        vmin=0, vmax=7000, alpha=0.5)
    plot_extent_box(axs[0, 0], extent_na, edgecolor="c", linewidth=2)
    axs[0, 0].scatter(
        na_lons, na_lats, 
        c=na_capes, cmap=scatter_cmap, s=10, 
        vmin=0, vmax=7000, alpha=0.5)
    plot_extent_box(axs[0, 0], extent_au, edgecolor="m", linewidth=2)
    s = axs[0, 0].scatter(
        au_lons, au_lats, 
        c=au_capes, cmap=scatter_cmap, s=10, 
        vmin=0, vmax=7000, alpha=0.5)
    
    axs[0, 1].set_extent(extent_na)
    egh.healpix_show(daily_max_cape_97, ax=axs[0, 1], cmap='Reds')
    axs[0, 1].scatter(
        na_lons, na_lats,
        c=na_capes, cmap=scatter_cmap, s=100, 
        vmin=0, vmax=7000, alpha=0.5)
    
    axs[1, 0].set_extent(extent_sa)
    egh.healpix_show(daily_max_cape_97, ax=axs[1, 0], cmap='Reds')
    axs[1, 0].scatter(
        sa_lons, sa_lats, 
        c=sa_capes, cmap=scatter_cmap, s=100, 
        vmin=0, vmax=7000, alpha=0.5)

    axs[1, 1].set_extent(extent_au)
    egh.healpix_show(daily_max_cape_97, ax=axs[1, 1], cmap='Reds')
    axs[1, 1].scatter(
        au_lons, au_lats, 
        c=au_capes, cmap=scatter_cmap, s=100, 
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
    print("Saudi Arabia CAPE maxima:")
    for event in sa_cape_maxima:
        print(f"Date: {event['date']}, Time: {event['timestamp']}, Lon: {event['lon']:.2f}, Lat: {event['lat']:.2f}, CAPE: {event['cape']:.1f}")

    print("North America CAPE maxima:")
    for event in na_cape_maxima:
        print(f"Date: {event['date']}, Time: {event['timestamp']}, Lon: {event['lon']:.2f}, Lat: {event['lat']:.2f}, CAPE: {event['cape']:.1f}")

    print("Australia CAPE maxima:")
    for event in au_cape_maxima:
        print(f"Date: {event['date']}, Time: {event['timestamp']}, Lon: {event['lon']:.2f}, Lat: {event['lat']:.2f}, CAPE: {event['cape']:.1f}")
# %%
