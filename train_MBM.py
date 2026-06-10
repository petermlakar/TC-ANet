import torch
import torch.nn as nn
import torch.jit as jit

from dataset.dataset import Dataset, load_reforecast_train, load_reforecast_valid, reforecast_standardize, load_reforecast_data

from datetime import datetime, UTC
from os.path import join, exists, isfile
from os import mkdir, makedirs, rename
import json

from typing import Dict, Callable, Optional, Tuple
from copy import deepcopy

from tqdm import tqdm
import sys
import random

from torch.profiler import profile, record_function, ProfilerActivity

from models.MBM import Model

torch.autograd.set_detect_anomaly(True)

config: Dict[str, str] = json.load(open(join(sys.argv[1], "config.json"), "r"))

device: str = 'cuda:0'
dataset_path: str = config['dataset_path'] 
output_path: str = config['output_path'] 
experiment_name: str = f"{config['train_experiment_name']}_{int(datetime.now(UTC).timestamp())}_{random.randint(0, 1000)}"
#experiment_name: str = f"mbm_wind_{int(datetime.now(UTC).timestamp())}_{random.randint(0, 1000)}"
output_path: str = join(output_path, experiment_name)

write_to_file: bool = True 

if not exists(output_path) and write_to_file:
    makedirs(output_path)

learning_rate: float = 1e-3
weight_decay: float = 1e-8
batch_size: int = 256 
number_of_epochs: int = 1000
tolerance: int = 10

input_lead_times: int = 20
output_lead_times: int = 20

prefix: str = config['variable']

m, dataset_train = load_reforecast_train(dataset_path, load_reforecast_data, batch_size, device, prefix)
_, dataset_valid = load_reforecast_valid(dataset_path, load_reforecast_data, batch_size, device, prefix)
_, _, _, ysd = reforecast_standardize(dataset_valid, dataset_train, prefix)
reforecast_standardize(dataset_train, dataset_train, prefix)
number_of_stations: int = dataset_train.x.shape[0]

match prefix:
    case 'temperature':
        target_field_index: int = 0
    case 'precipitation':
        target_field_index: int = 22
    case 'wind':
        target_field_index: int = 21

model: Model = Model(target_field_index = target_field_index,
                     number_of_forecast_fields = 29,

                     number_of_lead_times = input_lead_times,
                     number_of_stations = number_of_stations,

                     nembeddings_per_lead_time = 10,
                     nembeddings_per_station = 20,

                     number_of_features = 128).to(device)

optim = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

def process_dataset(dataset: Dataset) -> float:

    loss: float = 0.0
    counter: int = 0

    for i in tqdm(range(len(dataset))):

        j, s, d, _, x, y = dataset[i]

        if prefix == 'precipitation' and model.training:
            thr: torch.Tensor = (y.exp() - 1)*ysd.squeeze()*1000
            y = y.clone()
            y[thr >= 40] = torch.nan

        _idx: torch.Tensor = (~y.isnan()).all(dim = (-2, -1))

        s = s[_idx]
        d = d[_idx]
        x = x[_idx]
        y = y[_idx]

        if x.shape[0] == 0:
            continue

        counter += 1

        loss_batch: torch.Tensor = model.loss(x, d, s, y)

        if model.training:

            optim.zero_grad()
            loss_batch.backward()
            optim.step()

        loss += loss_batch.item()

    return loss/counter

losses_train = []
losses_valid = []

best_valid_loss_checkpoint: int = 0
best_valid_loss_model: Model = None

for e in range(number_of_epochs):

    print(f"\n###############\nEpoch {e + 1}/{number_of_epochs}")

    model.train()
    dataset_train.shuffle()

    loss_train = process_dataset(dataset_train)
    model.eval()

    with torch.no_grad():
        loss_valid = process_dataset(dataset_valid)

    losses_train.append(loss_train)
    losses_valid.append(loss_valid)

    print(f"Train loss: {loss_train:.3f} Valid loss: {loss_valid:.3f}")
    print(f"Current best valid loss: {losses_valid[best_valid_loss_checkpoint]:.3f}")

    if loss_valid <= losses_valid[best_valid_loss_checkpoint]:
       
        print(f"New best model found...saving and creating checkpoint...")

        model_path = join(output_path, "model")

        best_valid_loss_checkpoint = e
        best_valid_loss_model = deepcopy(model)

        if write_to_file:

            checkpoint_path = join(output_path, f"checkpoint_{e + 1}_{loss_valid:3f}")
            mkdir(checkpoint_path)
            jit.save(jit.script(model.cpu()), join(checkpoint_path, "model"))

    if write_to_file:

        jit.save(jit.script(model.cpu()), join(output_path, "model_last"))

        with open(join(output_path, "losses_train.txt"), "w") as f:
            json.dump(losses_train, f)
        with open(join(output_path, "losses_valid.txt"), "w") as f:
            json.dump(losses_valid, f)

    model = model.to(device)

    if (e - best_valid_loss_checkpoint) >= tolerance:
        break        

