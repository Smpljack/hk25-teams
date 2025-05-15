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
    # Find the maximum number of cells across all datasets
    max_cells = max(data.sizes['cell'] for data in data_list)
    icell = np.arange(max_cells)
    # Pad datasets with fewer cells using NaN values
    padded_data_list = []
    for data in data_list:
        if data.sizes['cell'] < max_cells:
            # Create a new dataset with the same structure but padded with NaNs
            padded_data = data.pad(cell=(0, max_cells - data.sizes['cell']), 
                                 mode='constant', 
                                 constant_values=np.nan)
            padded_data_list.append(padded_data)
        else:
            padded_data_list.append(data)
    data_list = padded_data_list
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
# extent_sa = [35, 60, 10, 35]  # Saudi Arabia
extent_na = [-110, -70, 30, 55]
icell, region = cm.get_healpix_region(extent_na, zoom)
ds = ds.isel(cell=icell)
cape_maxima = cm.find_regional_cape_maxima(
    ds.CAPE_max,
    n_cases=n_cases_per_region)
region_extents_for_events = [
    get_region_extent_for_lat_lon(
        event.lat.values, event.lon.values, regional_radius)
    for event in cape_maxima.event]
#%%
fig, ax = plt.subplots(
    figsize=(10, 10), subplot_kw={"projection": ccrs.PlateCarree()})
ax.set_extent(extent_na)
ax.scatter(cape_maxima.lon, cape_maxima.lat, c='r', s=100)
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
ds_coarse_events = [ds_coarse.isel(cell=icell)
                    for icell in icell_for_event_regions]
ds_coarse_events = [get_temporal_slice(
ds_coarse_event, event.time.values, 
ndays_around_event)
for ds_coarse_event, event in zip(
    ds_coarse_events, cape_maxima)]
#%%
combined_data = combine_temporal_slices(ds_coarse_events, cape_maxima.time)

#%%
event_index = 2
ta_anomaly = (combined_data.ta.isel(event=event_index).mean('icell') - 
              combined_data.ta.isel(event=event_index).mean('icell').mean('itime')).T
fig, ax = plt.subplots(figsize=(10, 10))
ax.pcolormesh(ta_anomaly.rel_time.values.astype('timedelta64[h]'), 
              ta_anomaly.plev.values, ta_anomaly)
ax.vlines(combined_data.isel(event=event_index).event_time,
          ta_anomaly.plev.max(), ta_anomaly.plev.min(), color='r')
ax.invert_yaxis()
    
# %%
