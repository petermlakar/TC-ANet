import torch
import torch.jit as jit

from dataset.dataset import Dataset, load_reforecast_train, load_reforecast_valid, load_reforecast_test, reforecast_standardize, load_forecast_data, load_reforecast_data
from evaluation.metrics import energy_score, crps_expectation 

from datetime import datetime, UTC
from os.path import join, exists, isfile, isdir
from os import mkdir, makedirs, rename, listdir
import json

from typing import Dict, Callable, List, Tuple, Optional

from tqdm import tqdm
import sys

from time import time

import matplotlib.pyplot as plt

from torch.profiler import profile, record_function, ProfilerActivity

config: Dict[str, str] = json.load(open(join(sys.argv[1], "config.json"), "r"))

device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dataset_path: str = config["dataset_path"]
output_path: str = config["output_path"]
input_path: str = output_path
experiment_name: str = config["predict_experiment_name"]
dataset_type: str = config['datasetType']

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

m, dataset_train = load_reforecast_train(dataset_path, data_loader, batch_size, device, prefix, target_field_index)

match dataset_type:
    
    case "train":
        xmu, xsd, mu, sd = reforecast_standardize(dataset_train, dataset_train, prefix)
        dataset = dataset_train
    case "valid":
        _, dataset_valid = load_reforecast_valid(dataset_path, data_loader, batch_size, device, prefix, target_field_index)
        xmu, xsd, mu, sd = reforecast_standardize(dataset_valid, dataset_train, prefix)
        dataset = dataset_valid
    case "test":
        _, dataset_test = load_reforecast_test(dataset_path, data_loader, batch_size, device, prefix, target_field_index)
        xmu, xsd, mu, sd = reforecast_standardize(dataset_test, dataset_train, prefix)
        dataset = dataset_test
    case "testf":
        _, dataset_test = load_forecast_data(dataset_path, batch_size, device, prefix)
        xmu, xsd, mu, sd = reforecast_standardize(dataset_test, dataset_train, prefix)
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

model_weights_path: str = join(input_path, 'model_weights.pt')
if exists(model_weights_path):
    model_weights: torch.Tensor = torch.tensor(torch.load(model_weights_path, weights_only = True))
else:
    model_weights: torch.Tensor = torch.ones((len(models_regression),), dtype = torch.float32)/len(models_regression)

model_weights = model_weights.to(device)

([print(k) for k in listdir(input_path) if isdir(join(input_path, k))])

# # # # #

prediction_type: str = config['predictionType']

number_of_steps: int = 100
number_of_samples: int = number_of_members//len(models_regression)
number_of_quantiles: int = number_of_members

match prediction_type:
    case 'marginal':
        output: torch.Tensor = torch.zeros((number_of_stations, number_of_run_times, number_of_lead_times, number_of_quantiles, number_of_variables), dtype = torch.float32)*torch.nan
    case 'joint':
        output: torch.Tensor = torch.zeros((number_of_stations, number_of_run_times, number_of_lead_times, number_of_members, number_of_variables), dtype = torch.float32)*torch.nan
    case _:
        raise RuntimeError(f'Invalid prediction type {prediction_type}')

mu, sd = mu.squeeze(), sd.squeeze()

def generate_samples_marginal(x: torch.Tensor, 
                              d: torch.Tensor, 
                              s: torch.Tensor,
                              embeddings: Optional[List[torch.Tensor]] = None) -> torch.Tensor:

    samples: List[torch.Tensor] = []

    for n, model in enumerate(models_regression):

        embedding: torch.Tensor = model.get_embeddings(x, d, s) if embeddings is None else embeddings[n]
        samples.append(model.model_marginal.sample_marginal_quantile(number_of_quantiles, model.init_marginal(embedding)))

    samples: torch.Tensor = torch.stack(samples, dim = 0)#.mean(dim = 0)

    samples = (model_weights[:, None, None, None]*samples).sum(dim = 0)

    return samples.detach().cpu()

def generate_samples_joint(x: torch.Tensor, 
                           d: torch.Tensor, 
                           s: torch.Tensor,
                           embeddings: Optional[List[torch.Tensor]] = None) -> torch.Tensor:

    samples: List[torch.Tensor] = []
    sampler: torch.Tensor = torch.randint(0, len(models_regression), size = (number_of_members % len(models_regression),))

    for n, model in enumerate(models_regression):

        embedding: torch.Tensor = model.get_embeddings(x, d, s) if embeddings is None else embeddings[n]

        N: int = number_of_samples + (sampler == n).sum().item()

        match prefix:
            case 'temperature':
                samples.append(model.sample_joint(N, number_of_steps, embedding).contiguous())
            case 'precipitation' | 'wind':
                samples.append(model.sample_joint(N, 
                                                  number_of_steps, 
                                                  embedding).contiguous().prod(dim = -1)[..., None])

    samples: torch.Tensor = torch.cat(samples, dim = -2)

    return samples.detach().cpu()

def generate_samples_joint_ecc(x: torch.Tensor, 
                               d: torch.Tensor, 
                               s: torch.Tensor,
                               forecast_ecc: bool = False) -> torch.Tensor:

    embeddings: List[torch.Tensor] = [model.get_embeddings(x, d, s) for model in models_regression]

    marginal: torch.Tensor = generate_samples_marginal(x, d, s, embeddings)

    if forecast_ecc:
        match prefix:
            case 'temperature':
                idx: torch.Tensor = x[..., 0].argsort(dim = -1).argsort(dim = -1).cpu()
            case 'precipitation':
                idx: torch.Tensor = x[..., 22].argsort(dim = -1).argsort(dim = -1).cpu()
            case 'wind':
                idx: torch.Tensor = x[..., 21].argsort(dim = -1).argsort(dim = -1).cpu()
    else:
        joint: torch.Tensor = generate_samples_joint(x, d, s, embeddings)[..., 0]
        idx: torch.Tensor = joint.argsort(dim = -1).argsort(dim = -1)

    return torch.gather(marginal, -1, idx)[..., None]

for i in tqdm(range(len(dataset))):
    
    j, s, d, td, x, y = dataset[i]

    with torch.no_grad():

        match prediction_type:
            case 'joint':
                samples: torch.Tensor = generate_samples_joint_ecc(x, d, s, False)
            case 'marginal':
                samples: torch.Tensor = generate_samples_marginal(x, d, s)[..., None]

    match prefix:
        case 'temperature':
            samples = samples*sd + mu
        case 'precipitation':
            samples[samples < 0.0] = 0.0
            samples = (samples.exp() - 1.0)*sd
        case 'wind':
            samples[samples < 0.0] = 0.0
            samples = (samples.exp() - 1.0)*sd
        case _:
            raise RuntimeError(f'Denormalization not implemented for this variable {prefix}')


    output[j[0], j[1]] = samples

torch.save(output, join(output_path, f'predictions_{prediction_type}_{dataset_type}_{int(time())}'))


