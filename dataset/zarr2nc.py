import numpy
import xarray

from os.path import exists, join
from os import mkdir

countries = ['belgium', 'austria', 'france', 'germany', 'netherlands']
pressure_levels = [500, 700, 850]
base = 'EUPPBench-stations'

for country in countries:

    if not exists(country):
        mkdir(country)

    path = join('.', f'{country}')

    print(f'Acquiring {country} data...')

    reforecast_surface = xarray.open_zarr(join(base, f'stations_ensemble_reforecasts_surface_{country}.zarr'))
    reforecast_surface_observations = xarray.open_zarr(join(base, f'stations_reforecasts_observations_surface_{country}.zarr'))

    forecast_surface = xarray.open_zarr(join(base, f'stations_ensemble_forecasts_surface_{country}.zarr'))
    forecast_surface_observations = xarray.open_zarr(join(base, f'stations_forecasts_observations_surface_{country}.zarr'))

    forecast_surface_observations.to_netcdf(join(path, 'observations_forecast_surface.nc'))
    reforecast_surface_observations.to_netcdf(join(path, 'observations_reforecast_surface.nc'))

    forecast_surface.to_netcdf(join(path, 'forecast_surface.nc'))
    reforecast_surface.to_netcdf(join(path, 'reforecast_surface.nc'))

    #### #### #### #### #### #### ####

    forecast_surface_processed = xarray.open_zarr(join(base, f'stations_ensemble_forecasts_surface_postprocessed_{country}.zarr'))
    reforecast_surface_processed = xarray.open_zarr(join(base, f'stations_ensemble_reforecasts_surface_postprocessed_{country}.zarr'))

    forecast_surface_processed_observations = xarray.open_zarr(join(base, f'stations_forecasts_observations_surface_postprocessed_{country}.zarr'))
    reforecast_surface_processed_observations = xarray.open_zarr(join(base, f'stations_reforecasts_observations_surface_postprocessed_{country}.zarr'))

    forecast_surface_processed.to_netcdf(join(path, 'forecast_processed.nc'))
    reforecast_surface_processed.to_netcdf(join(path, 'reforecast_processed.nc'))

    forecast_surface_processed_observations.to_netcdf(join(path, 'observations_forecast_processed.nc'))
    reforecast_surface_processed_observations.to_netcdf(join(path, 'observations_reforecast_processed.nc'))

    #### #### #### #### #### #### ####

    for pressure in pressure_levels:

        forecast_pressure = xarray.open_zarr(join(base, f'stations_ensemble_forecasts_pressure_{pressure}_{country}.zarr'))
        reforecast_pressure = xarray.open_zarr(join(base, f'stations_ensemble_reforecasts_pressure_{pressure}_{country}.zarr'))

        forecast_pressure.to_netcdf(join(path, f'forecast_pressure_{pressure}.nc'))
        reforecast_pressure.to_netcdf(join(path, f'reforecast_pressure_{pressure}.nc'))

