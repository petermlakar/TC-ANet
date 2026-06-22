import torch
import torch.jit as jit

from dataset.dataset import Dataset, load_reforecast_train, load_reforecast_valid, load_reforecast_test, reforecast_standardize, load_forecast_data, load_reforecast_data

from datetime import datetime, UTC
from os.path import join, exists, isfile, isdir
from os import mkdir, makedirs, rename, listdir
import json

from typing import Dict, Callable, List

from tqdm import tqdm
import sys

from time import time


config: Dict[str, str] = json.load(open(join(sys.argv[1], "config.json"), "r"))

device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dataset_path: str = config["dataset_path"]
output_path: str = config["output_path"]
input_path: str = output_path
experiment_name: str = config["predict_experiment_name"]
dataset_type: str = config['dataset_type']

input_path: str = join(input_path, experiment_name)
output_path: str = join(output_path, experiment_name)

batch_size: int = 2048

data_loader: Callable = load_reforecast_data

prefix: str = config['variable'] 

match prefix:
    case 'temperature':
        censored: bool = False
    case 'precipitation' | 'wind':
        censored: bool = True
    case _:
        censored: bool = False

m, dataset_train = load_reforecast_train(dataset_path, data_loader, batch_size, device, prefix)

match dataset_type:
    
    case "train":
        xmu, xds, mu, sd = reforecast_standardize(dataset_train, dataset_train, prefix)
        dataset = dataset_train
    case "valid":
        _, dataset_valid = load_reforecast_valid(dataset_path, data_loader, batch_size, device, prefix)
        xmu, xds, mu, sd = reforecast_standardize(dataset_valid, dataset_train, prefix)
        dataset = dataset_valid
    case "test":
        _, dataset_test = load_reforecast_test(dataset_path, data_loader, batch_size, device, prefix)
        xmu, xsd, mu, sd = reforecast_standardize(dataset_test, dataset_train, prefix)
        dataset = dataset_test
    case "testf":
        _, dataset_test = load_forecast_data(dataset_path, batch_size, device, prefix)
        ymu, xsd, mu, sd = reforecast_standardize(dataset_test, dataset_train, prefix)
        dataset = dataset_test
    case _:
        raise RuntimeError(f'Invalid dataset type {dataset_type}')

dataset.prediction = True

number_of_stations: int = dataset.x.shape[0]
number_of_run_times: int = dataset.x.shape[1]
number_of_lead_times: int = dataset.x.shape[2]
number_of_members: int = dataset.x.shape[3]
number_of_variables: int = dataset.y.shape[-1]

models_regression: List[torch.nn.Module] = [jit.load(join(input_path, k, "model"), map_location = torch.device("cpu")).to(device) for k in listdir(input_path) if isdir(join(input_path, k))]
[model.eval() for model in models_regression]

([print(k) for k in listdir(input_path) if isdir(join(input_path, k))])

prediction_type: str = config['prediction_type']

number_of_quantiles: int = number_of_members

resample_count: int = 1 if dataset_type == 'testf' else 1

match prediction_type:
    case 'marginal':
        output: torch.Tensor = torch.zeros((number_of_stations, number_of_run_times, number_of_lead_times, number_of_quantiles, number_of_variables), dtype = torch.float32)*torch.nan
    case 'joint':
        output: torch.Tensor = torch.zeros((number_of_stations, number_of_run_times, number_of_lead_times, number_of_members, number_of_variables), dtype = torch.float32)*torch.nan
    case _:
        raise RuntimeError(f'Invalid prediction type {prediction_type}')

mu, sd = mu.squeeze(), sd.squeeze()

#importance_file: str = 'importance_idx_precipitation'
#importance_index: torch.Tensor = torch.load(importance_file, weights_only = True).abs().squeeze().argsort(descending = True).to('cpu')
#dataset.x = dataset.x[..., importance_index]

with torch.no_grad():

    for i in tqdm(range(len(dataset))):
        
        j, s, d, td, x, y = dataset[i]
        samples: List[torch.Tensor] = []

        if prediction_type == 'joint':
            sampler: torch.Tensor = torch.randint(0, len(models_regression), size = (number_of_members % len(models_regression),))

        for n, model in enumerate(models_regression):

            for k in range(resample_count):

                match prediction_type:
                    case 'marginal':
                        samples.append(model.model_marginal.sample_marginal_quantile(number_of_quantiles, model(x, d, s))[..., None])

                    case 'joint':
                        samples.append(model.model_marginal.sample_marginal_quantile(number_of_quantiles, model(x, d, s)))

        match prediction_type:
            case 'marginal':
                samples: torch.Tensor = torch.stack(samples, dim = -1).mean(dim = -1)
            case 'joint':
                samples: torch.Tensor = torch.stack(samples, dim = -1).mean(dim = -1)

                match prefix:
                    case 'temperature':
                        _f_idx: torch.Tensor = x[..., 0].argsort(dim = -1).argsort(dim = -1)
                    case 'precipitation':
                        _f_idx: torch.Tensor = x[..., 22].argsort(dim = -1).argsort(dim = -1)
                    case 'wind':
                        _f_idx: torch.Tensor = x[..., 21].argsort(dim = -1).argsort(dim = -1)

                samples = torch.gather(samples, -1, _f_idx)[..., None]

        samples = samples.detach().cpu()

        match prefix:
            case 'temperature':
                samples = samples*sd + mu
            case 'precipitation' | 'wind':
                samples[samples < 0.0] = 0.0
                samples = (samples.exp() - 1.0)*sd
            case _:
                raise RuntimeError(f'Denormalization not implemented for this variable {prefix}')

        output[j[0], j[1]] = samples

torch.save(output, join(output_path, f"predictions_{prediction_type}_{dataset_type}_{int(time())}"))

