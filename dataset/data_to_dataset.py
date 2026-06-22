import torch
from os.path import join
from os import mkdir

import numpy as np
import netCDF4 as nc

from datetime import datetime, UTC
import json

station_index = 0
station_metadata = {}

# 

_forecasts_training = None
_forecasts_test = None

_observations_training = None
_observations_test = None

_time_training = None
_time_test = None

for country in ['austria', 'belgium', 'france', 'germany', 'netherlands']:

    print('Processing: ', country)

    forecast_surface   = nc.Dataset(join('.', country, 'forecast_surface.nc'))
    reforecast_surface = nc.Dataset(join('.', country, 'reforecast_surface.nc'))

    # Reforecasts hours since 1997-01-02

    observations_forecast_surface   = nc.Dataset(join('.', country, 'observations_forecast_surface.nc'))
    observations_reforecast_surface = nc.Dataset(join('.', country, 'observations_reforecast_surface.nc'))

    observations_forecast_processed = nc.Dataset(join('.', country, 'observations_forecast_processed.nc'))
    observations_reforecast_processed = nc.Dataset(join('.', country, 'observations_reforecast_processed.nc'))

    station_variables = ['station_id', 'station_name', 'longitude', 'latitude', 'altitude']
    for id, name, lon, lat, alt in zip(*[observations_forecast_surface[k][:] for k in station_variables]):
        station_metadata[id.item()] = {'name': name, 'lon': lon.item(), 'lat': lat.item(), 'alt': alt.item(), 'index': station_index, 'country': country}
        station_index += 1

    forecast_processed   = nc.Dataset(join('.', country, 'forecast_processed.nc'))
    reforecast_processed = nc.Dataset(join('.', country, 'reforecast_processed.nc'))

    forecast_850 = nc.Dataset(join('.', country, 'forecast_pressure_850.nc'))
    reforecast_850 = nc.Dataset(join('.', country, 'reforecast_pressure_850.nc'))

    forecast_700 = nc.Dataset(join('.', country, 'forecast_pressure_700.nc'))
    reforecast_700 = nc.Dataset(join('.', country, 'reforecast_pressure_700.nc'))

    forecast_500 = nc.Dataset(join('.', country, 'forecast_pressure_500.nc'))
    reforecast_500 = nc.Dataset(join('.', country, 'reforecast_pressure_500.nc'))

    # Temperature variables
    dataset_prefix: str = 'temperature'
    sfc_var  = ['t2m', 'u10', 'v10', 'tcc', 'u100', 'v100', 'tcw', 'tcwv', 'sd', 'cape', 'stl1', 'swvl1', 'vis']
    sfcp_var = ['ssrd6', 'ssr6', 'sshf6', 'slhf6', 'str6', 'strd6', 'mn2t6', 'mx2t6', 'p10fg6', 'tp6', 'cp6']
    p850_var = ['t']
    p700_var = ['q', 'u', 'v']
    p500_var = ['z']

    #sfc_var  = ['t2m', 'stl1', 'u10', 'v10', 'u100', 'v100']
    #sfcp_var = ['strd6', 'mn2t6', 'mx2t6', 'p10fg6']
    #p850_var = ['t']
    #p700_var = ['u', 'v']
    #p500_var = []

    # t2m, mx2t6, mn2t6, stl1, p10fg6, strd6

    # Precipitation variables
    #dataset_prefix: str = 'precipitation'
    #sfc_var = ['tcw', 'tcwv', 'u10', 'v10']
    #sfcp_var = ['tp6', 'cp6']
    #p850_var = []
    #p700_var = []
    #p500_var = ['z']

    obs_sfc_var = ['t2m']
    obs_prc_var = ['tp6', 'p10fg6']

    d = datetime.strptime('1997-01-02', '%Y-%m-%d').replace(tzinfo = UTC).timestamp()
    dp = datetime.strptime('1997-01-02T06:00:00', '%Y-%m-%dT%H:%M:%S').replace(tzinfo = UTC).timestamp()
    dpt = datetime.strptime('2017-01-01T06:00:00', '%Y-%m-%dT%H:%M:%S').replace(tzinfo = UTC).timestamp()

    # Skip first lead time which corresponds to the analysis run

    t_training = [reforecast_surface['valid_time'][:][..., 1:]*60**2 + d, reforecast_processed['valid_time'][:]*60**2 + dp, reforecast_850['valid_time'][:][..., 1:]*60**2 + d, reforecast_700['valid_time'][:][..., 1:]*60**2 + d]
    t_test     = [forecast_surface['valid_time'][:][..., 1:], forecast_processed['valid_time'][:]*60**2 + dpt, forecast_850['valid_time'][:][..., 1:], forecast_700['valid_time'][:][..., 1:]]

    for i in range(1, len(t_training)):
        assert (t_training[i - 1] == t_training[i]).all()
        assert (t_test[i - 1] == t_test[i]).all()

    t_training = torch.tensor(t_training[0], dtype = torch.int64).flatten(0, 1)
    t_test = torch.tensor(t_test[0], dtype = torch.int64)

    forecast_test = []
    forecast_training = []

    for f in sfc_var:
        forecast_training.append(torch.tensor(reforecast_surface[f][:], dtype = torch.float32)[..., 1:, 0])
        forecast_test.append(torch.tensor(forecast_surface[f][:], dtype = torch.float32)[..., 1:, 0])

        forecast_training[-1][torch.isnan(forecast_training[-1])] = -1
        forecast_test[-1][torch.isnan(forecast_test[-1])] = -1

    for f in sfcp_var:
        forecast_training.append(torch.tensor(reforecast_processed[f][:], dtype = torch.float32)[..., 0])
        forecast_test.append(torch.tensor(forecast_processed[f][:], dtype = torch.float32)[..., 0])
    for f in p850_var:
        forecast_training.append(torch.tensor(reforecast_850[f][:], dtype = torch.float32)[..., 1:, 0])
        forecast_test.append(torch.tensor(forecast_850[f][:], dtype = torch.float32)[..., 1:, 0])
    for f in p700_var:
        forecast_training.append(torch.tensor(reforecast_700[f][:], dtype = torch.float32)[..., 1:, 0])
        forecast_test.append(torch.tensor(forecast_700[f][:], dtype = torch.float32)[..., 1:, 0])

        # Replace NaN values with -0.1
        forecast_training[-1][torch.isnan(forecast_training[-1])] = 0

    for f in p500_var:
        forecast_training.append(torch.tensor(reforecast_500[f][:], dtype = torch.float32)[..., 1:, 0])
        forecast_test.append(torch.tensor(forecast_500[f][:], dtype = torch.float32)[..., 1:, 0])

        # Replace NaN values with -0.1
        forecast_training[-1][torch.isnan(forecast_training[-1])] = 0

    forecast_test = torch.stack(forecast_test, dim = -1).swapaxes(1, 2).swapaxes(-3, -2)
    forecast_training = torch.stack(forecast_training, dim = -1).swapaxes(-3, -4).swapaxes(-3, -2).flatten(1, 2)
  
    observations_training = []
    observations_test = []

    for f in obs_sfc_var:
        observations_training.append(torch.tensor(observations_reforecast_surface[f][:][None].swapaxes(0, -1)[..., 1:, 0], dtype = torch.float32).flatten(1, 2))
        observations_test.append(torch.tensor(observations_forecast_surface[f][:][None].swapaxes(0, -1)[..., 1:, 0], dtype = torch.float32))

    for f in obs_prc_var:
        observations_training.append(torch.tensor(observations_reforecast_processed[f][:][None].swapaxes(0, -1)[..., 0], dtype = torch.float32).flatten(1, 2))
        observations_test.append(torch.tensor(observations_forecast_processed[f][:][None].swapaxes(0, -1)[..., 0], dtype = torch.float32))

    observations_training = torch.stack(observations_training, dim = -1)
    observations_test = torch.stack(observations_test, dim = -1)

    _observations_training = observations_training if _observations_training is None else torch.cat([_observations_training, observations_training], dim = 0)
    _observations_test = observations_test if _observations_test is None else torch.cat([_observations_test, observations_test], dim = 0)

    _forecasts_training = forecast_training if _forecasts_training is None else torch.cat([_forecasts_training, forecast_training], dim = 0)
    _forecasts_test = forecast_test if _forecasts_test is None else torch.cat([_forecasts_test, forecast_test], dim = 0)

    assert (~(_forecasts_training.isnan())).all()
    assert (~(_forecasts_test.isnan())).all()

    if _time_training is None:
        _time_training = t_training
    else:
        assert (_time_training == t_training).all()

    if _time_test is None:
        _time_test = t_test
    else:
        assert (_time_test == t_test).all()

print(_forecasts_training.shape, _observations_training.shape)
print(_forecasts_test.shape, _observations_test.shape)

torch.save(_forecasts_training, 'forecasts_training.dst')
torch.save(_forecasts_test, 'forecasts_test.dst')

torch.save(_observations_training, 'observations_training.dst')
torch.save(_observations_test, 'observations_test.dst')

torch.save(_time_training, 'time_training.dst')
torch.save(_time_test, 'time_test.dst')

with open('station_metadata.json', 'w') as f:
    json.dump(station_metadata, f)

