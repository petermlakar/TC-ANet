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

device: str = "cuda:0" if torch.cuda.is_available() else 'cpu'
dataset_path: str = config["dataset_path"]
output_path: str = config["output_path"]
input_path: str = output_path
experiment_name: str = config["predict_experiment_name"]
dataset_type: str = config['datasetType']

input_path: str = join(input_path, experiment_name)
output_path: str = join(output_path, experiment_name)

batch_size: int = 2048

data_loader: Callable = load_reforecast_data

prefix: str = 'wind'#config['variable']

m, dataset_train = load_reforecast_train(dataset_path, data_loader, batch_size, device, prefix)

match dataset_type:
    
    case "train":
        xmu, xsd, mu, sd = reforecast_standardize(dataset_train, dataset_train, prefix, residuals = False)
        dataset = dataset_train
    case "valid":
        _, dataset_valid = load_reforecast_valid(dataset_path, data_loader, batch_size, device, prefix, None)
        xmu, xsd, mu, sd = reforecast_standardize(dataset_valid, dataset_train, prefix, residuals = False)
        dataset = dataset_valid
    case "test":
        _, dataset_test = load_reforecast_test(dataset_path, data_loader, batch_size, device, prefix, None)
        xmu, xsd, mu, sd = reforecast_standardize(dataset_test, dataset_train, prefix, residuals = False)
        dataset = dataset_test
    case "testf":
        _, dataset_test = load_forecast_data(dataset_path, batch_size, device, prefix)
        xmu, xsd, mu, sd = reforecast_standardize(dataset_test, dataset_train, prefix, residuals = False)
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

print(len(models_regression))

prediction_type: str = config['predictionType']

match prediction_type:
    case 'joint' | 'marginal':
        output: torch.Tensor = torch.zeros((number_of_stations, number_of_run_times, number_of_lead_times, number_of_members, number_of_variables), dtype = torch.float32)*torch.nan
    case _:
        raise RuntimeError(f'Invalid prediction type {prediction_type}')

mu, sd = mu.squeeze(), sd.squeeze()

with torch.no_grad():

    for i in tqdm(range(len(dataset))):
        
        j, s, d, td, x, y = dataset[i]
        samples: List[torch.Tensor] = []

        for n, model in enumerate(models_regression):

            match prediction_type:
                case 'marginal' | 'joint':
                    samples.append(model(x, d, s).contiguous())
    
        match prediction_type:
            case 'marginal' | 'joint':
                samples: torch.Tensor = torch.stack(samples, dim = -1).mean(dim = -1)
        samples = samples.detach().cpu()

        match prefix:
            case 'temperature':
                samples = samples*sd + mu
            case 'wind' | 'precipitation':
                samples[samples < 0.0] = 0.0
                samples = (samples.exp() - 1.0)*sd
            case _:
                raise RuntimeError(f'Denormalization not implemented for this variable {prefix}')

        output[j[0], j[1]] = samples[..., None]
        #output[j[0], j[1]] = output[j[0], j[1]][..., torch.rand(number_of_members*len(models_regression)).argsort(), :]


torch.save(output, join(output_path, f"predictions_{prediction_type}_{dataset_type}_{int(time())}"))

