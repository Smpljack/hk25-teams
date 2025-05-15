#%%
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import healpix as hp
import easygems.healpix as egh
import intake     # For catalogs

import cape_maps as cm

def get_region_extent_for_lat_lon(lat, lon, km_radius):
    # Convert km to degrees
    km_to_dlat = 1/111.32
    km_to_dlon = 1/111.32 * np.cos(np.deg2rad(lat))
    # Convert radius to degrees
    dlat = km_radius * km_to_dlat
    dlon = km_radius * km_to_dlon
    # Get extent
    extent = [lon - dlon, lon + dlon, 
              lat - dlat, lat + dlat]
    return extent

def get_temporal_slice(data, event_time, ndays_around_event):
    time_slice = slice(event_time - np.timedelta64(ndays_around_event, 'D'),
                       event_time + np.timedelta64(ndays_around_event, 'D'))
    event_data = data.sel(time=time_slice)
    event_time_deltas = (event_data.time.values - event_time)
    event_data = event_data.assign_coords(
        rel_time=("time", event_time_deltas))
    # Create integer index for the time dimension
    itime = np.arange(event_data.sizes['time'])
    event_data = event_data.assign_coords(itime=("time", itime))
    event_data = event_data.swap_dims({"time": "itime"})
    event_data = event_data.reset_coords('time')
    return event_data

def combine_temporal_slices(data_list, event_times):
    # Convert lat/lon coordinates to variables with event dimension
    icell = np.arange(len(data_list[0].cell))
    lats = xr.DataArray(
        data=np.array([event.lat.values for event in data_list]),
        coords={'event': range(len(data_list)), 'icell': icell},
        dims=['event', 'icell']
    )
    lons = xr.DataArray(
        data=np.array([event.lon.values for event in data_list]),
        coords={'event': range(len(data_list)), 'icell': icell},
        dims=['event', 'icell']
    )
    cells = xr.DataArray(
        data=np.array([event.cell.values for event in data_list]),
        coords={'event': range(len(data_list)), 'icell': icell},
        dims=['event', 'icell']
    )
    data_list = [data.rename_dims({'cell': 'icell'}).drop_vars('cell') for data in data_list]
    for data in data_list:
        data['icell'] = icell

    combined_data = xr.concat(data_list, dim='event')
    combined_data = combined_data.reset_coords(['lat', 'lon'])
    combined_data['lat'] = lats
    combined_data['lon'] = lons
    combined_data['cell'] = cells
    combined_data['event_time'] = ("event", event_times.data)
    return combined_data

#%%
# if __name__ == "__main__":
zoom = 7
n_cases_per_region = 20
regional_radius = 100
ndays_around_event = 7
ds = cm.load_native_xsh24_data(zoom=zoom).sel(
    time=slice('2020-01-01', '2020-12-31'))
# Select region
extent_sa = [35, 60, 10, 35]  # Saudi Arabia
icell, region = cm.get_healpix_region(extent_sa, zoom)
ds_sa = ds.isel(cell=icell)
sa_cape_maxima = cm.find_regional_cape_maxima(
    ds_sa.CAPE_max,
        n_cases=n_cases_per_region)
region_extents_for_events = [
    get_region_extent_for_lat_lon(
        event.lat.values, event.lon.values, regional_radius)
    for event in sa_cape_maxima.event]
#%%
fig, ax = plt.subplots(
    figsize=(10, 10), subplot_kw={"projection": ccrs.PlateCarree()})
ax.set_extent(extent_sa)
ax.scatter(sa_cape_maxima.lon, sa_cape_maxima.lat, c='r', s=100)
ax.coastlines()
for extent in region_extents_for_events:
    cm.plot_extent_box(ax, extent)
# for extent in region_extents_for_events:
#     ax.add_patch(plt.Polygon(extent, fill=False, color='red'))
# %%
ds_coarse = cm.load_coarse_xsh24_data(zoom=zoom)
ds_coarse['time'] = ds_coarse.time.values.astype('datetime64[h]')
icell_for_event_regions = [cm.get_healpix_region(
extent, zoom)[0] for extent in region_extents_for_events]
ds_coarse_sa_events = [ds_coarse.isel(cell=icell)
                    for icell in icell_for_event_regions]
ds_coarse_sa_events = [get_temporal_slice(
ds_coarse_sa_event, event.time.values, 
ndays_around_event)
for ds_coarse_sa_event, event in zip(
    ds_coarse_sa_events, sa_cape_maxima)]
#%%
combined_data = combine_temporal_slices(ds_coarse_sa_events, sa_cape_maxima.time)

#%%
fig, ax = plt.subplots(
    figsize=(10, 10), subplot_kw={"projection": ccrs.PlateCarree()})
ds_coarse_sa_events
    
# %%
