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

    def set_batch_size(self, batch_size: int):

        self.length = math.ceil(self.indices.shape[-1]/batch_size) 
        self.batch_size = batch_size

    def __getitem__(self, i: int):
       
        if not self.importance_sampling:

            i0 = i*self.batch_size
            i1 = min(self.indices.shape[1], (i + 1)*self.batch_size)

            i: torch.Tensor = self.indices[:, i0:i1] 

        else:

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

            y[37] = torch.nan

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

        case "wind":
            y = y[..., 2][..., None]
            y[y < 0.0] = 0.0
        case "joint":
            pass

    return m, (t, x, y), (idx_train, idx_valid, idx_test), (_idx_train, _idx_valid, _idx_test)

def load_reforecast_train(path: str,
                          loader: Callable[str, str],
                          batch_size: int = 256,
                          device: str = "cpu",
                          prefix: str = "temperature") -> Tuple[Dict, Dataset]:

    m, (t, x, y), (_, idx_valid, _), (_idx_train, _, _) = loader(path, prefix)

    t_train, x_train, y_train = t[_idx_train].clone(), x[:, _idx_train].clone(), y[:, _idx_train].clone()
    y_train[:, idx_valid[_idx_train]] = torch.nan

    print('Train data: ', x_train.shape, y_train.shape)

    return m, Dataset(t_train, x_train, y_train, batch_size = batch_size, device = device)

def load_reforecast_valid(path: str,
                          loader: Callable[str, str],
                          batch_size: int = 256,
                          device: str = "cpu",
                          prefix: str = "temperature") -> Tuple[Dict, Dataset]:
    
    m, (t, x, y), (_, _, idx_test), (_, _idx_valid, _) = loader(path, prefix)

    t_valid, x_valid, y_valid = t[_idx_valid].clone(), x[:, _idx_valid].clone(), y[:, _idx_valid].clone()
    y_valid[:, idx_test[_idx_valid]]  = torch.nan

    print('Valid data: ', x_valid.shape, y_valid.shape)

    return m, Dataset(t_valid, x_valid, y_valid, batch_size = batch_size, device = device)

def load_reforecast_test(path: str,
                         loader: Callable[str, str],
                         batch_size: int = 256,
                         device: str = "cpu",
                         prefix: str = "temperature") -> Tuple[Dict, Dataset]:
    
    m, (t, x, y), (_, _, _), (_, _, _idx_test) = loader(path, prefix)

    t_test,  x_test,  y_test  = t[_idx_test],  x[:, _idx_test],  y[:, _idx_test]

    print('Test data: ', x_test.shape, y_test.shape)

    return m, Dataset(t_test, x_test, y_test, batch_size = batch_size, device = device)

def reforecast_standardize(target: Dataset, source: Dataset, prefix: str = "temperature") -> Tuple[torch.Tensor, torch.Tensor]:

    xmu, xsd = source.x.mean(dim = (0, 1, 2, 3), keepdim = True), source.x.std(dim = (0, 1, 2, 3), keepdim = True)

    ymu: torch.Tensor = source.y.nanmean(dim = (0, 1, 2), keepdim = True)
    ysd: torch.Tensor = (source.y - ymu).pow(2).nanmean(dim = (0, 1, 2), keepdim = True).sqrt()

    target.x = (target.x - xmu)/xsd

    match prefix:
        case 'temperature':
            target.y = (target.y - ymu)/ysd
        case 'precipitation' | 'wind':
            target.y = (target.y - ymu)/ysd

        case _:
            print(f"Dataset {prefix} standardization not supported")
            exit()

    return xmu, xsd, ymu, ysd

