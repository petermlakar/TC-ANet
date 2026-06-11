import torch
import numpy as np

from os.path import join

import json
import math

from datetime import datetime, UTC
from typing import Tuple, Dict, List, Optional, Callable

from copy import deepcopy

class Dataset:

    def __init__(self, 
                 t: torch.Tensor,
                 x: torch.Tensor,
                 y: torch.Tensor,
                 batch_size: int = 256,
                 device: str = "cpu"):

        stations, run, time, ensemble, variables = x.shape

        self.station_indices = torch.arange(stations, dtype = torch.int32)

        self.t, self.x, self.y = t, x, y
        self.indices = torch.stack(torch.meshgrid(torch.arange(stations), torch.arange(run), indexing = "ij"), dim = 0).flatten(1)

        doy = torch.tensor(np.vectorize(lambda k: datetime.fromtimestamp(k, UTC).timetuple().tm_yday)(t), dtype = torch.float32)/23.0
        tod = torch.tensor(np.vectorize(lambda k: datetime.fromtimestamp(k, UTC).hour)(t), dtype = torch.float32)
        self.doy = doy/doy.max()
        self.tod = tod

        self.length = math.ceil(self.indices.shape[-1]/batch_size) 
        self.batch_size = batch_size
        self.device = device

        self.prediction = False
        self.shuffle_members = False
        self.reduce_members = False
        self.importance_sampling = False

        # Compute obervations weights

    def set_batch_size(self, batch_size: int):

        self.length = math.ceil(self.indices.shape[-1]/batch_size) 
        self.batch_size = batch_size

    def set_importance_sampling(self, variable: str, weights: Optional[torch.Tensor] = None):

        self.importance_sampling = True 

        match variable:

            case 'precipitation' | 'wind':

                weights: List[torch.Tensor] = []

                for i in range(self.y.shape[0]):

                    #daily_prec: torch.Tensor = (self.y[i, ..., 0] > 0).sum(dim = -1).type(torch.float32)
                    #daily_prec: torch.Tensor = self.x[i, ..., 22].mean(dim = (-1, -2))
                    daily_prec: torch.Tensor = self.y[i, ..., 0].nanmean(dim = -1)

                    hv, he = torch.histogram(daily_prec[~daily_prec.isnan()], bins = torch.arange(20, dtype = torch.float32, device = daily_prec.device))
                    hv = 1/(hv + 1e-8)

                    daily_prec_shape: torch.Size = daily_prec.shape
                    _weights: torch.Tensor = hv[(daily_prec[:, None] >= he[1:][None]).sum(dim = 1).clamp(max = len(hv) - 1)].view(daily_prec_shape)
                    _weights /= _weights.sum()
                    weights.append(_weights.cumsum(dim = -1))

                self.weights: torch.Tensor = torch.stack(weights)
             
            case 'wind':
                pass

    def __getitem__(self, i: int):
       
        if not self.importance_sampling:

            i0 = i*self.batch_size
            i1 = min(self.indices.shape[1], (i + 1)*self.batch_size)

            i: torch.Tensor = self.indices[:, i0:i1] 

        else:

            #_i0: int = int(self.batch_size*0.2)
            #_i1: int = self.batch_size - _i0
            #i_stations_zero: torch.Tensor = self.importance_distribution_zero_stations[torch.randint(low = 0, high = len(self.importance_distribution_zero_stations), size = (_i0,))]
            #i_stations_posi: torch.Tensor = self.importance_distribution_posi_stations[torch.randint(low = 0, high = len(self.importance_distribution_posi_stations), size = (_i1,))]
            #i_run_time_zero: torch.Tensor = (self.importance_i0[i_stations_zero] < torch.rand((_i0, 1))).sum(dim = -1).clamp(min = 0, max = self.y.shape[1] - 1)
            #i_run_time_posi: torch.Tensor = (self.importance_i1[i_stations_posi] < torch.rand((_i1, 1))).sum(dim = -1).clamp(min = 0, max = self.y.shape[1] - 1)
            #i: torch.Tensor = torch.stack([torch.cat([i_stations_zero, i_stations_posi]), torch.cat([i_run_time_zero, i_run_time_posi])], dim = 0)[:, torch.rand(self.batch_size).argsort()]

            i_stations: torch.Tensor = torch.randint(low = 0, high = self.y.shape[0], size = (self.batch_size,))
            i_run_time: torch.Tensor = (self.weights[i_stations] < torch.rand((len(i_stations), 1))).sum(dim = -1).clamp(min = 0, max = self.y.shape[1] - 1)
            i: torch.Tensor = torch.stack([i_stations, i_run_time], dim = 0)

        doy = self.doy[i[1]]
        t = self.tod[i[1]]
        x = self.x[i[0], i[1]]
        y = self.y[i[0], i[1]]
        s = self.station_indices[i[0]]

        if self.shuffle_members:

            k = torch.rand(x.shape[-2]).argsort()
            x = x[..., k, :]

        if self.reduce_members:

            k = torch.rand(x.shape[-2]).argsort()[:torch.randint(low = int(x.shape[-2]*0.6), high = x.shape[-2] + 1, size = (1,))[0]]
            x = x[..., k, :]
            
        if self.device.startswith('cuda'):
            
            doy = doy.to(self.device)
            t = t.to(self.device)
            x = x.to(self.device)
            y = y.to(self.device)
            s = s.to(self.device)

        return (i, s, doy, t, x, y)

    def __len__(self):
        return self.length

    def shuffle(self):
        self.indices = self.indices[:, torch.argsort(torch.rand(self.indices.shape[-1]))]

def load_data(path: str, postfix: str = "training", prefix: str = "temperature") -> Tuple[Dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    t: torch.Tensor = torch.load(join(path, f"time_{postfix}_temperature.dst"), weights_only = False)
    x: torch.Tensor = torch.load(join(path, f"forecasts_{postfix}_temperature.dst"), weights_only = False)
    y: torch.Tensor = torch.load(join(path, f"observations_{postfix}_temperature.dst"), weights_only = False)

    return json.load(open(join(path, "station_metadata.json"), "r")), t, x, y

def load_metadata(path: str) -> Dict:
    return json.load(open(join(path, "station_metadata.json"), "r"))

# # # #

def load_forecast_data(path: str,
                       batch_size: int = 1,
                       device: str = "cpu",
                       prefix: str = "temperature"):
    
    m, t, x, y = load_data(path, "test", prefix)

    match prefix:

        case "temperature":
            y = y[..., 0][..., None]
        case "precipitation":
            y = y[..., 1][..., None]
            y[y < 0.0] = 0.0

            # Set station 30 observations to NaN since
            # the training dataset has no recorded precipitation
            # for that station
            #y[30] = torch.nan
            #y[20] = torch.nan

            y[37] = torch.nan
            #y[y*1000 < 0.2] = 0.0

        case "wind":
            y = y[..., 2][..., None]

    return m, Dataset(t, x, y, batch_size = batch_size, device = device)

def load_reforecast_data(path: str,
                         prefix: str = "temperature"):

    m, t, x, y = load_data(path, "training", prefix)

    idx_train: torch.Tensor = torch.tensor(np.vectorize(lambda k: datetime.fromtimestamp(k, UTC).year <= 2009)(t), dtype = torch.bool)
    idx_valid: torch.Tensor = torch.tensor(np.vectorize(lambda k: datetime.fromtimestamp(k, UTC).year >= 2010 and datetime.fromtimestamp(k, UTC).year <= 2013)(t), dtype = torch.bool)
    idx_test:  torch.Tensor = torch.tensor(np.vectorize(np.vectorize(lambda k: datetime.fromtimestamp(k, UTC).year >= 2014 and datetime.fromtimestamp(k, UTC).year <= 2017))(t), dtype = torch.bool)
    
    _idx_train: torch.Tensor = idx_train[:, 0]
    _idx_valid: torch.Tensor = idx_valid[:, 0]
    _idx_test:  torch.Tensor = idx_test[:, 0]

    match prefix:

        case "temperature":
            y = y[..., 0][..., None]
        case "precipitation":
            y = y[..., 1][..., None]
            y[y < 0.0] = 0.0
            # Set station 37 observations to NaN since
            # the training dataset has no recorded precipitation
            # for that station
            y[37] = torch.nan
            #y[y*1000 < 0.2] = 0.0

        case "wind":
            y = y[..., 2][..., None]
            y[y < 0.0] = 0.0
        case "joint":
            pass

    return m, (t, x, y), (idx_train, idx_valid, idx_test), (_idx_train, _idx_valid, _idx_test)

def apply_residuals(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    print('Residuals -> ', x.shape, y.shape)
    return y - x.mean(dim = -1, keepdim = True)

def load_reforecast_train(path: str,
                          loader: Callable[str, str],
                          batch_size: int = 256,
                          device: str = "cpu",
                          prefix: str = "temperature",
                          residuals: Optional[int] = None) -> Tuple[Dict, Dataset]:

    m, (t, x, y), (_, idx_valid, _), (_idx_train, _, _) = loader(path, prefix)

    t_train, x_train, y_train = t[_idx_train].clone(), x[:, _idx_train].clone(), y[:, _idx_train].clone()
    y_train[:, idx_valid[_idx_train]] = torch.nan

    print('Train data: ', x_train.shape, y_train.shape)

    if not (residuals is None):
        y_train = apply_residuals(x_train[..., residuals], y_train)

    return m, Dataset(t_train, x_train, y_train, batch_size = batch_size, device = device)

def load_reforecast_valid(path: str,
                          loader: Callable[str, str],
                          batch_size: int = 256,
                          device: str = "cpu",
                          prefix: str = "temperature",
                          residuals: Optional[int] = None) -> Tuple[Dict, Dataset]:
    
    m, (t, x, y), (_, _, idx_test), (_, _idx_valid, _) = loader(path, prefix)

    t_valid, x_valid, y_valid = t[_idx_valid].clone(), x[:, _idx_valid].clone(), y[:, _idx_valid].clone()
    y_valid[:, idx_test[_idx_valid]]  = torch.nan

    print('Valid data: ', x_valid.shape, y_valid.shape)

    if not (residuals is None):
        y_valid = apply_residuals(x_valid[..., residuals], y_valid)

    return m, Dataset(t_valid, x_valid, y_valid, batch_size = batch_size, device = device)

def load_reforecast_test(path: str,
                         loader: Callable[str, str],
                         batch_size: int = 256,
                         device: str = "cpu",
                         prefix: str = "temperature",
                         residuals: Optional[int] = None) -> Tuple[Dict, Dataset]:
    
    m, (t, x, y), (_, _, _), (_, _, _idx_test) = loader(path, prefix)

    t_test,  x_test,  y_test  = t[_idx_test],  x[:, _idx_test],  y[:, _idx_test]

    print('Test data: ', x_test.shape, y_test.shape)

    if not (residuals is None):
        y_test = apply_residuals(x_test[..., residuals], y_test)

    return m, Dataset(t_test, x_test, y_test, batch_size = batch_size, device = device)

def reforecast_standardize(target: Dataset, source: Dataset, prefix: str = "temperature", residuals: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:

    xmu, xsd = source.x.mean(dim = (0, 1, 2, 3), keepdim = True), source.x.std(dim = (0, 1, 2, 3), keepdim = True)

    ymu: torch.Tensor = source.y.nanmean(dim = (0, 1, 2), keepdim = True)
    ysd: torch.Tensor = (source.y - ymu).pow(2).nanmean(dim = (0, 1, 2), keepdim = True).sqrt()

    target.x = (target.x - xmu)/xsd

    match prefix:
        case 'temperature':
            target.y = (target.y - ymu)/ysd
        case 'precipitation' | 'wind':

            if not residuals:
                target.y[target.y < 0.0] = 0.0
                target.y = torch.log(target.y/ysd + 1.0)

            else:
                target.y = (target.y - ymu)/ysd

        case _:
            print(f"Dataset {prefix} standardization not supported")
            exit()

    return xmu, xsd, ymu, ysd

