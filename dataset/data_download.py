import numpy
import climetlab as cml

from os.path import exists, join
from os import mkdir

countries = ["belgium", "austria", "france", "germany", "netherlands"]
pressure_levels = [500, 700, 850]


for country in countries:

    if not exists(country):
        mkdir(country)
    else:
        continue

    path = join(".", f"{country}")

    print(f"Acquiring {country} data...")

    forecast_surface = cml.load_dataset('EUPPBench-training-data-stations-forecasts-surface', "ensemble", country)
    reforecast_surface = cml.load_dataset("EUPPBench-training-data-stations-reforecasts-surface", country)

    forecast_surface.get_observations_as_xarray().to_netcdf(join(path, f"observations_forecast_surface.nc"))
    reforecast_surface.get_observations_as_xarray().to_netcdf(join(path, f"observations_reforecast_surface.nc"))

    forecast_surface.to_xarray().to_netcdf(join(path, f"forecast_surface.nc"))
    reforecast_surface.to_xarray().to_netcdf(join(path, f"reforecast_surface.nc"))

    #### #### #### #### #### #### ####

    forecast_processed = cml.load_dataset("EUPPBench-training-data-stations-forecasts-surface-processed", "ensemble", country)
    reforecast_processed = cml.load_dataset("EUPPBench-training-data-stations-reforecasts-surface-processed", country)

    forecast_processed.get_observations_as_xarray().to_netcdf(join(path, f"observations_forecast_processed.nc"))
    reforecast_processed.get_observations_as_xarray().to_netcdf(join(path, f"observations_reforecast_processed.nc"))

    forecast_processed.to_xarray().to_netcdf(join(path, f"forecast_processed.nc"))
    reforecast_processed.to_xarray().to_netcdf(join(path, f"reforecast_processed.nc"))

    #### #### #### #### #### #### ####

    for pressure in pressure_levels:

        forecast_pressure = cml.load_dataset("EUPPBench-training-data-stations-forecasts-pressure", pressure, "ensemble", country).to_xarray()
        reforecast_pressure = cml.load_dataset("EUPPBench-training-data-stations-reforecasts-pressure", pressure, country).to_xarray()
        forecast_pressure.to_netcdf(join(path, f"forecast_pressure_{pressure}.nc"))
        reforecast_pressure.to_netcdf(join(path, f"reforecast_pressure_{pressure}.nc"))

