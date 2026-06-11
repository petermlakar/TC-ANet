import torch
import torch.jit as jit
import numpy as np

from os.path import join, exists
from os import listdir, mkdir, makedirs

import json

from datetime import datetime, UTC
import sys

import matplotlib.pyplot as plt

from evaluation.metrics import diebold_mariano, twcrps, tbrier, binary_reliability, coverage, roc, brier_score, crps_ensemble, crps_quantile_decomposition, crps_expectation, quantile_loss, rank_histogram, sharpness_composite, energy_score, variogram_score, variogram_create_locality_matrix
from evaluation.plotting_functions import plot_stratify, plot_map, plot_score, plot_reliability, plot_skill_score, plot_roc, plot_crps, plot_energy_per_y, plot_vario_per_y, plot_rank, plot_rank_per_station, plot_case, plot_distribution 

from dataset.dataset import Dataset, load_reforecast_train, load_reforecast_valid, load_reforecast_test, reforecast_standardize, load_reforecast_data, load_forecast_data

from typing import List, Tuple, Dict, Callable, Union, Optional
from tqdm import tqdm

print('Started eval...')

config: Dict[str, str] = json.load(open("config.json" if len(sys.argv) == 1 else sys.argv[1], "r"))

prefix: str = config['variable']

dataset_path: str = config["dataset_path"]
output_path: str = join(config["evaluation_path"], f"{int(datetime.now(UTC).timestamp())}_{prefix}")
model_path: str = config["output_path"]
dataset_type: str = config['datasetType']

station_metadata: Dict = json.load(open(join(dataset_path, 'station_metadata.json'), 'r'))

data_loader: Callable = load_reforecast_data

print('Loading dataset...', end = '')

match dataset_type:
    case 'train':
        m, dataset = load_reforecast_train(dataset_path, data_loader, 1, 'cpu', prefix)
    case 'valid':
        m, dataset = load_reforecast_valid(dataset_path, data_loader, 1, 'cpu', prefix)
    case 'test':
        m, dataset = load_reforecast_test(dataset_path, data_loader, 1, 'cpu', prefix)
    case 'testf':
        m, dataset = load_forecast_data(dataset_path, 1, 'cpu', prefix)
print('Done!\n')

t, x, y = dataset.t, dataset.x, dataset.y

colors: List[Tuple[float, float, float]] = [(103, 161, 85),
                                            (85, 161, 161),
                                            (161, 133, 85),
                                            (161, 85, 96),
                                            (156, 156, 95)]
colors = [(c[0]/255, c[1]/255, c[2]/255) for c in colors]

markers: str = ["o", "v", "^", "h", "X", "D"]

match prefix:
    case 'temperature':
        y = y[..., 0]
    case 'precipitation':
        y[y < 0.0] = 0.0
        y = y[..., 0]*1000
    case 'wind':
        y = y[..., 0]
        y[y < 0.0] = 0.0
    case _:
        raise RuntimeError('Invalid prefix ', prefix)

y_lim: torch.Tensor = torch.quantile(y[~y.isnan()], q = torch.tensor([0.1, 0.99]))

match prefix:
    case 'temperature':
        var_name: str = 'Temperature'
        var_unit: str = '[$K$]'
        var_ylabel: str = 'Temperature [$K$]'
        var_ylim: Tuple[int, int] = (y_lim[0].round(), 290)

    case 'precipitation':
        var_name: str = 'Precipitation'
        var_unit: str = '[$mm$]'
        var_ylabel: str = 'Precipitation [$mm$]'
        var_ylim: Tuple[int, int] = (0, 40)

    case 'wind':
        var_name: str = 'Wind gusts'
        var_unit: str = r'[$\frac{m}{s}$]'
        var_ylabel: str = r'Wind gusts [$\frac{m}{s}$]'
        var_ylim: Tuple[int, int] = (0, 60)

models: List[Dict] = [{"type": mdl["type"], 
                       "label": mdl["label"], 
                       'modelPath': join(config['output_path'], mdl['modelFolder']),
                       'colorIndex': mdl['colorIndex'],
                       'markerIndex': mdl['markerIndex'], 
                       'predictions': None,
                       'predictionType': mdl['predictionType']} for mdl in config["evaluation_models"]]

models.append({'type': 'ECMWF', 'label': 'ECMWF', 'model_path': None, 'predictions': None, 'colorIndex': 0, 'markerIndex': 0})

station_names: Dict[int, str] = {k["index"]: k["name"] for i, k in m.items()}

def load_models(model: Dict):

    print('Loading ', model['label'], end = '...')

    match model['type']:
        case 'ANet':

            model['predictions'] = [f for f in listdir(model['modelPath']) if f"predictions_{model['predictionType']}_{dataset_type}_" in f]

        case 'ECMWF':

            match prefix:
                case 'temperature':
                    model['predictions'] = [x[..., 0]]
                case 'precipitation':
                    model['predictions'] = [x[..., 22]*1000]
                case 'wind':
                    model['predictions'] = [x[..., 21]]
                case _:
                    raise RuntimeError('Invalid prefix ', prefix)

    model['color'] = colors[model['colorIndex']]
    model['marker'] = markers[model['markerIndex']]

    print('Done!')

    return model

models = [load_models(k) for k in models]

if not exists(output_path):
    makedirs(output_path)

## ## ## ## ## ## ## ## ## ##

blacklist_scores: List[str] = []
blacklist_crps: List[str] = []
blacklist_rank: List[str] = []
blacklist_rank_ps: List[str] = []
whitelist_case: List[str] = ["Combined", "ECMWF", "Joint"]

## ## ## ## ## ## ## ## ## ##
#y[y <= 0.0] = float('nan')

stats: Dict[str, Dict[str, float | List[float]]] = {}

#### #### #### #### ####

def get_per_level_scores(o: torch.Tensor, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    onan: torch.Tensor = ~(o.isnan())
    n: int = onan.sum()

    c: List[torch.Tensor] = []
    x: List[torch.Tensor] = []
    y: List[torch.Tensor] = []

    for v in o[onan].unique().sort()[0]:

        idx: torch.Tensor = torch.logical_and(onan, o >= v)

        c.append(idx.sum()/n)
        x.append(v)
        y.append(s[idx].nanmean())

    return torch.tensor(c), torch.tensor(x), torch.tensor(y)

#### #### #### #### ####

def nanstd(x: torch.Tensor, dim: Optional[Tuple[int]] = None) -> torch.Tensor:

    if not (dim is None):
        m: torch.Tensor = x.nanmean(dim = dim, keepdim = True)

        return (x - m).pow(2).nanmean(dim = dim).sqrt()

    return x[~x.isnan()].std()

def accumulate(model: Dict, var: str, detensify: bool = True):

    N: int = len(model[var])

    model[var] = torch.stack(model[var], dim = 0)
    if N > 1:
        model[f'{var}_acc_sd'] = model[var].std(dim = 0)
    model[var] = model[var].mean(dim = 0)

    if detensify:
        if len(model[var].shape) > 0:
            model[var] = model[var].tolist()

            if N > 1:
                model[f'{var}_acc_sd'] = model[f'{var}_acc_sd'].tolist()
        else:
            model[var] = model[var].item()

            if N > 1:
                model[f'{var}_acc_sd'] = model[f'{var}_acc_sd'].item()


ecmwf_baseline: torch.Tensor = [model['predictions'] for model in models if model['label'] == 'ECMWF'][0][0].sort(dim = -1)[0]

print((y[~y.isnan()] > 0).sum()/(y[~y.isnan()]).numel())

for model in models:

    if model['label'] in blacklist_scores:
        continue

    stats[model['label']] = {}

    match prefix:
        case 'temperature':

            print(f"Evaluating {model['label']} with {len(model['predictions'])}...")

            idx: torch.Tensor = (~y.isnan()).all(dim = -1).flatten()

            stats[model['label']]['crps'] = []
            stats[model['label']]['qloss'] = []
            stats[model['label']]['energy'] = []
            stats[model['label']]['crps_per_lead_time'] = []
            stats[model['label']]['sharpness50'] = []
            stats[model['label']]['sharpness95'] = []

            stats[model['label']]['crps_sd'] = []
            stats[model['label']]['qloss_sd'] = []
            stats[model['label']]['energy_sd'] = []
            stats[model['label']]['crps_per_lead_time_sd'] = []
            stats[model['label']]['sharpness50_sd'] = []
            stats[model['label']]['sharpness95_sd'] = []

            model['crps_per_station'] = []
            model['crpss'] = []
            model['ess'] = []

            model['crps'] = []
            model['energy'] = []
            model['qloss'] = []
            model['brier'] = []
            model['rank'] = []

            for i, predictions_file in enumerate(model['predictions']):

                predictions: torch.Tensor = torch.load(join(model['modelPath'], predictions_file), weights_only = True)[..., 0] if model['type'] != 'ECMWF' else predictions_file
                
                #predictions = predictions[..., torch.randperm(51)[:11]]

                predictions_sorted: torch.Tensor = predictions.sort(dim = -1)[0]


                print(predictions.shape)

                #score_crps: torch.Tensor = torch.cat([crps_expectation(p, _y) for (p, _y) in zip(torch.chunk(predictions_sorted.flatten(end_dim = -2), dim = 0, chunks = 1000), torch.chunk(y.flatten(end_dim = -1), dim = 0, chunks = 1000))], dim = 0).view(predictions_sorted.shape[:-1])
                #score_energy: torch.Tensor = torch.cat([energy_score(p, _y) for (p, _y) in zip(torch.chunk(predictions.flatten(end_dim = 1), dim = 0, chunks = 1000), torch.chunk(y.flatten(end_dim = 1), dim = 0, chunks = 1000))], dim = 0).view(predictions.shape[:-2])

                score_sharpness50: torch.Tensor = sharpness_composite(predictions_sorted, 0.5, True)
                score_sharpness95: torch.Tensor = sharpness_composite(predictions_sorted, 0.95, True)
                
                #score_crps: torch.Tensor = crps_expectation(predictions_sorted, y)
                #score_energy: torch.Tensor = energy_score(predictions, y)
                score_crps: torch.Tensor = torch.cat([crps_expectation(p, _y) for (p, _y) in zip(torch.chunk(predictions_sorted.flatten(end_dim = -2), dim = 0, chunks = 1000), torch.chunk(y.flatten(end_dim = -1), dim = 0, chunks = 1000))], dim = 0).view(predictions_sorted.shape[:-1])

                print(f'{score_crps.nanmean():.3f}')

                score_energy: torch.Tensor = torch.cat([energy_score(p, _y) for (p, _y) in zip(torch.chunk(predictions.flatten(end_dim = 1), dim = 0, chunks = 1000), torch.chunk(y.flatten(end_dim = 1), dim = 0, chunks = 1000))], dim = 0).view(predictions.shape[:-2])
                score_rank: torch.Tensor = rank_histogram(predictions_sorted, y, censored = True)
                score_quantile: torch.Tensor = quantile_loss(predictions_sorted, y)

                stats[model['label']]['crps'].append(score_crps.nanmean(dim = (0, 1, 2)))
                stats[model['label']]['qloss'].append(score_quantile.nanmean())
                stats[model['label']]['energy'].append(score_energy.nanmean())
                stats[model['label']]['crps_per_lead_time'].append(score_crps.nanmean(dim = (0, 1)))
                stats[model['label']]['sharpness50'].append(score_sharpness50.mean())
                stats[model['label']]['sharpness95'].append(score_sharpness95.mean())

                stats[model['label']]['crps_sd'].append(nanstd(score_crps, dim = (0, 1, 2)))
                stats[model['label']]['qloss_sd'].append(nanstd(score_quantile))
                stats[model['label']]['energy_sd'].append(nanstd(score_energy))
                stats[model['label']]['crps_per_lead_time_sd'].append(nanstd(score_crps, dim = (0, 1)))
                stats[model['label']]['sharpness50_sd'].append(score_sharpness50.std())
                stats[model['label']]['sharpness95_sd'].append(score_sharpness95.std())

                model['crps_per_station'].append(score_crps.nanmean(dim = (1, 2)))
                model['predictions'] = predictions
                model['crps'].append(score_crps)
                model['energy'].append(score_energy)
                model['rank'].append(score_rank)
                model['qloss'].append(score_quantile)

                #c_crpss, x_crpss, y_crpss = twcrps(predictions_sorted, y)

                #model['crpss'].append(y_crpss)
                #model['crpss_x'] = x_crpss
                #model['crpss_c'] = c_crpss

            accumulate(stats[model['label']], 'crps')
            accumulate(stats[model['label']], 'qloss')
            accumulate(stats[model['label']], 'energy')
            accumulate(stats[model['label']], 'crps_per_lead_time')
            accumulate(stats[model['label']], 'sharpness50')
            accumulate(stats[model['label']], 'sharpness95')
            accumulate(model, 'rank', detensify = False)

            accumulate(stats[model['label']], 'crps_sd')
            accumulate(stats[model['label']], 'qloss_sd')
            accumulate(stats[model['label']], 'energy_sd')
            accumulate(stats[model['label']], 'crps_per_lead_time_sd')
            accumulate(stats[model['label']], 'sharpness50_sd')
            accumulate(stats[model['label']], 'sharpness95_sd')

            model['crps_raw'] = [m.clone() for m in model['crps']]
            model['energy_raw'] = [m.clone() for m in model['energy']]

            model['crps'] = [m.nanmean(dim = (0, 1)) for m in model['crps']]
            model['qloss'] = [m.nanmean(dim = (0, 1, 2)) for m in model['qloss']]

            print(f'Crps: [{stats[model['label']]['crps']:.3f}] Qloss: [{stats[model['label']]['qloss']:.3f}] Energy: [{stats[model['label']]['energy']:.3f}]\n')

        case 'precipitation':

            print(f"Evaluating {model['label']} with {len(model['predictions'])}...")

            idx: torch.Tensor = (~y.isnan()).all(dim = -1).flatten()

            stats[model['label']]['crps'] = []
            stats[model['label']]['twcrps'] = []
            stats[model['label']]['brier'] = []
            stats[model['label']]['qloss'] = []
            stats[model['label']]['energy'] = []
            stats[model['label']]['crps_per_lead_time'] = []
            stats[model['label']]['sharpness50'] = []
            stats[model['label']]['sharpness95'] = []

            stats[model['label']]['crps_sd'] = []
            stats[model['label']]['twcrps_sd'] = []
            stats[model['label']]['brier_sd'] = []
            stats[model['label']]['qloss_sd'] = []
            stats[model['label']]['energy_sd'] = []
            stats[model['label']]['crps_per_lead_time_sd'] = []
            stats[model['label']]['sharpness50_sd'] = []
            stats[model['label']]['sharpness95_sd'] = []

            model['crps_per_station'] = []
            model['crpss'] = []
            model['tbrierss'] = []

            model['energy'] = []
            model['crps'] = []
            model['qloss'] = []
            model['brier'] = []
            model['rank'] = []

            for i, predictions_file in enumerate(model['predictions']):

                predictions: torch.Tensor = torch.load(join(model['modelPath'], predictions_file), weights_only = True)[..., 0]*1000 if model['type'] != 'ECMWF' else predictions_file
                #predictions = predictions[..., torch.randperm(51)[:11]]
                #predictions: torch.Tensor = torch.load(join(model['modelPath'], predictions_file), weights_only = True)[..., 0] if model['type'] != 'ECMWF' else predictions_file
                predictions_sorted: torch.Tensor = predictions.sort(dim = -1)[0]

                #predictions_sorted = torch.log(predictions_sorted + 0.01)
                #y = torch.log(y + 0.01)

                score_sharpness50: torch.Tensor = sharpness_composite(predictions_sorted, 0.5, True)
                score_sharpness95: torch.Tensor = sharpness_composite(predictions_sorted, 0.95, True) 

                score_reliability: [Tuple[float, torch.Tensor]] = [(q, binary_reliability(predictions_sorted, y, q)) for q in torch.quantile(y[y > 0], torch.tensor([0.01, 0.25, 0.75, 0.99]))]
                score_reliability = [(0.0, binary_reliability(predictions_sorted, y, 0.0))] + score_reliability
                score_crps: torch.Tensor = torch.cat([crps_expectation(p, _y) for (p, _y) in zip(torch.chunk(predictions_sorted.flatten(end_dim = -2), dim = 0, chunks = 1000), torch.chunk(y.flatten(end_dim = -1), dim = 0, chunks = 1000))], dim = 0).view(predictions_sorted.shape[:-1])

                print(f'{score_crps.nanmean():.3f}')

                score_energy: torch.Tensor = torch.cat([energy_score(p, _y) for (p, _y) in zip(torch.chunk(predictions.flatten(end_dim = 1), dim = 0, chunks = 1000), torch.chunk(y.flatten(end_dim = 1), dim = 0, chunks = 1000))], dim = 0).view(predictions.shape[:-2])
                score_coverage: torch.Tensor = coverage(predictions_sorted, y)
                score_brier: torch.Tensor = brier_score(predictions_sorted, y)
                score_rank: torch.Tensor = rank_histogram(predictions_sorted, y, censored = True)
                score_quantile: torch.Tensor = quantile_loss(predictions_sorted, y)

                for l in score_crps.nanmean(dim = (0, 1)):
                    print(f'{l:.3f} ', end = ' ')

                print()

                stats[model['label']]['crps'].append(score_crps.nanmean(dim = (0, 1, 2)))
                stats[model['label']]['brier'].append(score_brier.nanmean())
                stats[model['label']]['qloss'].append(score_quantile.nanmean())
                stats[model['label']]['energy'].append(score_energy.nanmean())
                stats[model['label']]['crps_per_lead_time'].append(score_crps.nanmean(dim = (0, 1)))
                stats[model['label']]['sharpness50'].append(score_sharpness50.mean())
                stats[model['label']]['sharpness95'].append(score_sharpness95.mean())

                stats[model['label']]['crps_sd'].append(nanstd(score_crps, dim = (0, 1, 2)))
                stats[model['label']]['brier_sd'].append(nanstd(score_brier))
                stats[model['label']]['qloss_sd'].append(nanstd(score_quantile))
                stats[model['label']]['energy_sd'].append(nanstd(score_energy))
                stats[model['label']]['crps_per_lead_time_sd'].append(nanstd(score_crps, dim = (0, 1)))
                stats[model['label']]['sharpness50_sd'].append(score_sharpness50.std())
                stats[model['label']]['sharpness95_sd'].append(score_sharpness95.std())

                model['energy'].append(score_energy)
                model['crps_per_station'].append(score_crps.nanmean(dim = (1, 2)))
                model['predictions'] = predictions
                model['crps'].append(score_crps)
                model['brier'].append(score_brier)
                model['rank'].append(score_rank)
                model['qloss'].append(score_quantile)
                model['reliability'] = score_reliability

                c_crpss, x_crpss, y_crpss = twcrps(predictions_sorted, y, 30, model['label'])

                model['crpss'].append(y_crpss)
                model['crpss_x'] = x_crpss
                model['crpss_c'] = c_crpss

                c_tbrier, x_tbrier, y_tbrier = tbrier(predictions_sorted, y, n = 30)

                model['tbrierss'].append(y_tbrier)
                model['tbrierss_x'] = x_tbrier
                model['tbrierss_c'] = c_tbrier

            accumulate(stats[model['label']], 'crps')
            accumulate(stats[model['label']], 'brier')
            accumulate(stats[model['label']], 'qloss')
            accumulate(stats[model['label']], 'energy')
            accumulate(stats[model['label']], 'crps_per_lead_time')
            accumulate(stats[model['label']], 'sharpness50')
            accumulate(stats[model['label']], 'sharpness95')

            accumulate(stats[model['label']], 'crps_sd')
            accumulate(stats[model['label']], 'brier_sd')
            accumulate(stats[model['label']], 'qloss_sd')
            accumulate(stats[model['label']], 'energy_sd')
            accumulate(stats[model['label']], 'crps_per_lead_time_sd')
            accumulate(stats[model['label']], 'sharpness50_sd')
            accumulate(stats[model['label']], 'sharpness95_sd')
            accumulate(model, 'rank', detensify = False)

            model['crps_raw'] = [m.clone() for m in model['crps']]
            model['energy_raw'] = [m.clone() for m in model['energy']]

            model['crps'] = [m.nanmean(dim = (0, 1)) for m in model['crps']]
            model['qloss'] = [m.nanmean(dim = (0, 1, 2)) for m in model['qloss']]
            model['brier'] = [m.nanmean(dim = (0, 1)) for m in model['brier']]

            print(f'Crps: [{stats[model['label']]['crps']:.3f}] Brier: [{stats[model['label']]['brier']:.3f}] Qloss: [{stats[model['label']]['qloss']:.3f}] Energy: [{stats[model['label']]['energy']:.3f}]\n')

        case 'wind':
           
            print(f"Evaluating {model['label']} with {len(model['predictions'])}...")

            idx: torch.Tensor = (~y.isnan()).all(dim = -1).flatten()

            stats[model['label']]['crps'] = []
            stats[model['label']]['brier'] = []
            stats[model['label']]['qloss'] = []
            stats[model['label']]['energy'] = []
            stats[model['label']]['crps_per_lead_time'] = []
            stats[model['label']]['sharpness50'] = []
            stats[model['label']]['sharpness95'] = []

            stats[model['label']]['crps_sd'] = []
            stats[model['label']]['brier_sd'] = []
            stats[model['label']]['qloss_sd'] = []
            stats[model['label']]['energy_sd'] = []
            stats[model['label']]['crps_per_lead_time_sd'] = []
            stats[model['label']]['sharpness50_sd'] = []
            stats[model['label']]['sharpness95_sd'] = []

            model['crps_per_station'] = []
            model['crpss'] = []
            model['energy'] = []
            model['tbrierss'] = []

            model['crps'] = []
            model['qloss'] = []
            model['brier'] = []
            model['rank'] = []

            for i, predictions_file in enumerate(model['predictions']):

                predictions: torch.Tensor = torch.load(join(model['modelPath'], predictions_file), weights_only = True)[..., 0] if model['type'] != 'ECMWF' else predictions_file
                #predictions = predictions[..., torch.randperm(51)[:11]]
                predictions_sorted: torch.Tensor = predictions.sort(dim = -1)[0]

                score_sharpness50: torch.Tensor = sharpness_composite(predictions_sorted, 0.5, True)
                score_sharpness95: torch.Tensor = sharpness_composite(predictions_sorted, 0.95, True) 

                score_reliability: [Tuple[float, torch.Tensor]] = [(q, binary_reliability(predictions_sorted, y, q)) for q in torch.quantile(y[y > 0], torch.tensor([0.01, 0.25, 0.75, 0.99]))]
                score_crps: torch.Tensor = torch.cat([crps_expectation(p, _y) for (p, _y) in zip(torch.chunk(predictions_sorted.flatten(end_dim = -2), dim = 0, chunks = 1000), torch.chunk(y.flatten(end_dim = -1), dim = 0, chunks = 1000))], dim = 0).view(predictions_sorted.shape[:-1])
                print(f'{score_crps.nanmean():.3f}')

                score_energy: torch.Tensor = torch.cat([energy_score(p, _y) for (p, _y) in zip(torch.chunk(predictions.flatten(end_dim = 1), dim = 0, chunks = 1000), torch.chunk(y.flatten(end_dim = 1), dim = 0, chunks = 1000))], dim = 0).view(predictions.shape[:-2])
                score_coverage: torch.Tensor = coverage(predictions_sorted, y)
                score_brier: torch.Tensor = brier_score(predictions_sorted, y)
                score_rank: torch.Tensor = rank_histogram(predictions_sorted, y, censored = True)
                score_quantile: torch.Tensor = quantile_loss(predictions_sorted, y)

                stats[model['label']]['crps'].append(score_crps.nanmean(dim = (0, 1, 2)))
                stats[model['label']]['brier'].append(score_brier.nanmean())
                stats[model['label']]['qloss'].append(score_quantile.nanmean())
                stats[model['label']]['energy'].append(score_energy.nanmean())
                stats[model['label']]['crps_per_lead_time'].append(score_crps.nanmean(dim = (0, 1)))
                stats[model['label']]['sharpness50'].append(score_sharpness50.mean())
                stats[model['label']]['sharpness95'].append(score_sharpness95.mean())

                stats[model['label']]['crps_sd'].append(nanstd(score_crps, dim = (0, 1, 2)))
                stats[model['label']]['brier_sd'].append(nanstd(score_brier))
                stats[model['label']]['qloss_sd'].append(nanstd(score_quantile))
                stats[model['label']]['energy_sd'].append(nanstd(score_energy))
                stats[model['label']]['crps_per_lead_time_sd'].append(nanstd(score_crps, dim = (0, 1)))
                stats[model['label']]['sharpness50_sd'].append(score_sharpness50.std())
                stats[model['label']]['sharpness95_sd'].append(score_sharpness95.std())

                model['crps_per_station'].append(score_crps.nanmean(dim = (1, 2)))
                model['reliability'] = score_reliability
                model['predictions'] = predictions
                model['crps'].append(score_crps)
                model['energy'].append(score_energy)
                model['brier'].append(score_brier)
                model['rank'].append(score_rank)
                model['qloss'].append(score_quantile)

                c_crpss, x_crpss, y_crpss = twcrps(predictions_sorted, y, 30, model['label']) #get_per_level_scores(y.flatten(), score_crps.flatten())

                model['crpss'].append(y_crpss)
                model['crpss_x'] = x_crpss
                model['crpss_c'] = c_crpss

                c_tbrier, x_tbrier, y_tbrier = tbrier(predictions_sorted, y, n = 30)

                model['tbrierss'].append(y_tbrier)
                model['tbrierss_x'] = x_tbrier
                model['tbrierss_c'] = c_tbrier

            accumulate(stats[model['label']], 'crps')
            accumulate(stats[model['label']], 'brier')
            accumulate(stats[model['label']], 'qloss')
            accumulate(stats[model['label']], 'energy')
            accumulate(stats[model['label']], 'crps_per_lead_time')
            accumulate(stats[model['label']], 'sharpness50')
            accumulate(stats[model['label']], 'sharpness95') 
            accumulate(model, 'rank', detensify = False)

            accumulate(stats[model['label']], 'crps_sd')
            accumulate(stats[model['label']], 'brier_sd')
            accumulate(stats[model['label']], 'qloss_sd')
            accumulate(stats[model['label']], 'energy_sd')
            accumulate(stats[model['label']], 'crps_per_lead_time_sd')
            accumulate(stats[model['label']], 'sharpness50_sd')
            accumulate(stats[model['label']], 'sharpness95_sd')

            model['crps_raw'] = [m.clone() for m in model['crps']]
            model['energy_raw'] = [m.clone() for m in model['energy']]

            model['crps'] = [m.nanmean(dim = (0, 1)) for m in model['crps']]
            model['qloss'] = [m.nanmean(dim = (0, 1, 2)) for m in model['qloss']]
            model['brier'] = [m.nanmean(dim = (0, 1)) for m in model['brier']]

            print(f'Crps: [{stats[model['label']]['crps']:.3f}] Brier: [{stats[model['label']]['brier']:.3f}] Qloss: [{stats[model['label']]['qloss']:.3f}] Energy: [{stats[model['label']]['energy']:.3f}]\n')


# Diebold-Mariano CRPS
for i in range(len(models)):

    model1: Dict = models[i]

    for j in range(i + 1, len(models)):

        model2: Dict = models[j]

        diebold_mariano(model1['crps_raw'][0], model2['crps_raw'][0], model1['label'], model2['label'])

print()
# Diebold-Mariano ES

for i in range(len(models)):

    model1: Dict = models[i]

    for j in range(i + 1, len(models)):

        model2: Dict = models[j]

        diebold_mariano(model1['energy_raw'][0], model2['energy_raw'][0], model1['label'], model2['label'])


## ## ## ## ## ## ## ## ## ##

with open(join(output_path, 'stats.json'), 'w') as f:
    json.dump(stats, f)

## ## ## ## ## ## ## ## ## ##

def _plot_stratify():

    _config: Dict = {'outputPath': output_path, 'varName': var_name}
    plot_stratify(models, _config, station_metadata)

def _plot_map():

    _config: Dict = {'outputPath': output_path, 'varName': var_name}
    plot_map(models, _config, station_metadata)

def _plot_reliability():

    _config: Dict = {'outputPath': output_path, 'outputName': 'reliability', 'varName': var_name.lower(), 'varUnit': var_unit}
    plot_reliability(models, _config)

def _plot_twcrpss(baseline: str = 'ECMWF', legend: bool = True):

    q999: torch.Tensor = y[y > 0].quantile(q = 0.999)

    _config: Dict = {'outputPath': output_path, 'outputName': f'crpss_y_{baseline}', 'q999': q999.item(), 'xlabel': f'{var_name} threshold {var_unit}', 'markevery': 10, 'ylabel': f'twCRPSS [%]', 'c': models[0]['crpss_c'], 'x': models[0]['crpss_x'], 'metric': 'crpss', 'base': baseline}
    #_config: Dict = {'outputPath': output_path, 'outputName': 'crpss_y', 'xlabel': f'{var_name} threshold {var_unit}', 'markevery': 10, 'ylabel': f'Outcome Weighed\nContinuous Ranked Probability Score', 'metric': 'crpss', 'base': 'ECMWF'}

    if baseline == 'ECMWF':
        plot_skill_score(models, _config, legend)
    else:
        plot_skill_score([model for model in models if model['label'] != 'ECMWF'], _config, legend)

def _plot_tbrier(baseline: str = 'ECMWF', legend: bool = True):

    q999: torch.Tensor = y[y > 0].quantile(q = 0.999)

    _config: Dict = {'outputPath': output_path, 'outputName': f'tbrierss_y_{baseline}', 'q999': q999.item(), 'xlabel': f'{var_name} threshold {var_unit}', 'markevery': 10, 'ylabel': f'BSS [%]', 'c': models[0]['tbrierss_c'], 'x': models[0]['tbrierss_x'], 'metric': 'tbrierss', 'base': baseline}
    #_config: Dict = {'outputPath': output_path, 'outputName': 'crpss_y', 'xlabel': f'{var_name} threshold {var_unit}', 'markevery': 10, 'ylabel': f'Outcome Weighed\nContinuous Ranked Probability Score', 'metric': 'crpss', 'base': 'ECMWF'}

    if baseline == 'ECMWF':
        plot_skill_score(models, _config, legend)
    else:
        plot_skill_score([model for model in models if model['label'] != 'ECMWF'], _config, legend)

def _plot_energy_per_y():

    _config: Dict = {'outputPath': output_path, 'outputName': 'ess', 'xlabel': f'Mean {var_name.lower()} threshold {var_unit}', 'markevery': 10, 'ylabel': 'Energy Skill Score [%]', 'c': models[0]['ess_c'], 'x': models[0]['ess_x'], 'metric': 'ess', 'base': 'ECMWF'}
    plot_skill_score(models, _config)
 
def _plot_vario_per_y():

    _config: Dict = {'outputPath': output_path, 'outputName': 'vss', 'xlabel': f'Mean {var_name.lower()} threshold {var_unit}', 'markevery': 10, 'ylabel': 'Variogram Skill Score [%]', 'c': models[0]['vss_c'], 'x': models[0]['vss_x'], 'metric': 'vss', 'base': 'ECMWF'}
    plot_skill_score(models, _config)
 
def _plot_crpss(baseline: str = 'ECMWF', legend: bool = True):

    _config: Dict = {'outputPath': output_path, 'outputName': f'crpss_{baseline}', 'xlabel': 'Lead times [6 hours]', 'ylabel': 'CRPSS [%]' if prefix == 'temperature' else None, 'metric': 'crps', 'base': baseline, 'title': var_name}

    if baseline == 'ECMWF':
        plot_skill_score(models, _config, legend)
    else:
        plot_skill_score([model for model in models if model['label'] != 'ECMWF'], _config, legend)

def _plot_rank():

    _config: Dict = {'outputPath': output_path, 'prefix': prefix, 'datasetType': dataset_type}
    plot_rank(models, _config)

def _plot_qss(baseline: str = 'ECMWF', legend: bool = True):
   
    q: int = [model for model in models if model['type'] == 'ECMWF'][0]['qloss'][0].shape[-1]
    x: torch.Tensor = torch.arange(1/(1 + q), 1, 1/(1 + q))

    _config: Dict = {'outputPath': output_path, 'outputName': f'qss_{baseline}', 'xlabel': 'Quantiles', 'ylabel': 'QSS [%]' if prefix == 'temperature' else None, 'markevery': 5, 'x': x, 'metric': 'qloss', 'base': baseline}

    if baseline == 'ECMWF':
        plot_skill_score(models, _config, legend)
    else:
        plot_skill_score([model for model in models if model['label'] != 'ECMWF'], _config, legend)

def _plot_bss():

    _config: Dict = {'outputPath': output_path, 'outputName': 'bss', 'xlabel': 'Lead time [6 hours]', 'ylabel': 'Brier Skill Score', 'metric': 'brier', 'base': 'ECMWF'}
    plot_skill_score(models, _config)

def _plot_case():

    _config: Dict = {'outputPath': output_path, 'prefix': prefix, 'datasetType': dataset_type, 'ylabel': var_ylabel, 'ylim': var_ylim}
    plot_case(models, _config, y, station_names, station_metadata)

def _plot_roc():

    _config: Dict = {'outputPath': output_path, 'prefix': prefix, 'datasetType': dataset_type, 'ylabel': var_ylabel, 'ylim': var_ylim}
    plot_roc(models, _config)

def _plot_distribution():

    _config: Dict = {'outputPath': output_path, 'prefix': prefix, 'datasetType': dataset_type, 'xlabel': var_ylabel}
    plot_distribution(models, y, _config)

## ## ## ## ## ## ## ## ## ##

eval_commands = [(True, _plot_crpss),
                 (True, lambda: _plot_crpss('BQN')),
                 (True, _plot_rank),
                 (True, _plot_stratify),
                 (True, _plot_map),
                 (True, lambda: _plot_qss(legend = False)),
                 (True, lambda: _plot_qss('BQN', legend = False)),
                 (not True and (prefix != 'temperature'), _plot_bss),
                 (True and (prefix != 'temperature'), _plot_twcrpss),
                 (True and (prefix != 'temperature'), lambda: _plot_tbrier(legend = False)),
                 (True and (prefix != 'temperature'), lambda: _plot_twcrpss('BQN')),
                 (True and (prefix != 'temperature'), lambda: _plot_tbrier('BQN', legend = False)),
                 (True and (prefix != 'temperature'), _plot_reliability),
                 (True, _plot_case),
                 (not True, _plot_distribution),
                 (not True, _plot_roc)] 

[c[1]() for c in eval_commands if c[0]]

