import torch
import torch.jit as jit

from dataset.dataset import Dataset, load_reforecast_train, load_reforecast_valid, reforecast_standardize, load_reforecast_data

from datetime import datetime, UTC
from os.path import join, exists, isfile
from os import mkdir, makedirs, rename
import json
from typing import Dict, Callable, Optional, Tuple, List
from copy import deepcopy

from tqdm import tqdm
import sys
import random

from models.FLOW_ENCODER import ForecastEncoder
from models.FLOW_MARGINAL import ModelMarginal, Student
from models.FLOW_JOINT import ModelJoint
from models.FLOW import Model

config: Dict[str, str] = json.load(open(join(sys.argv[1], "config.json"), "r"))
hyper: Dict[str, Dict[str, int]] = json.load(open(join(sys.argv[1], 'hyperparams.json'), 'r'))

device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dataset_path: str = config["dataset_path"] 
output_path: str = config["output_path"] 
experiment_name: str = f"{config['train_experiment_name']}_{int(datetime.now(UTC).timestamp())}_{random.randint(0, 1000)}"
output_path: str = join(output_path, experiment_name)

write_to_file: bool = False 

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

match prefix:
    case 'temperature':
        censored: bool = False
        target_field_index: int = None #0
        residuals: bool = not (target_field_index is None)
    case 'precipitation':
        censored: bool = True
        target_field_index: int = None
        residuals: bool = False
    case 'wind':
        censored: bool = True
        target_field_index: int = None #21
        residuals: bool = not (target_field_index is None)
    case _:
        censored: bool = False
        target_field_index: int = None
        residuals: bool = False

m, dataset_train = load_reforecast_train(dataset_path, load_reforecast_data, batch_size, device, prefix)
_, dataset_valid = load_reforecast_valid(dataset_path, load_reforecast_data, batch_size, device, prefix)

#dataset_train.set_importance_sampling(prefix)
#dataset_valid.set_importance_sampling(prefix)

reforecast_standardize(dataset_valid, dataset_train, prefix, residuals = residuals)
_, _, _, ysd = reforecast_standardize(dataset_train, dataset_train, prefix, residuals = residuals)
number_of_stations: int = dataset_train.x.shape[0]

forecast_encoder: ForecastEncoder = ForecastEncoder(number_of_forecast_fields = dataset_train.x.shape[-1],
                                                    
                                                    number_of_input_lead_times = input_lead_times,
                                                    number_of_output_lead_times = output_lead_times,

                                                    number_of_stations = number_of_stations,

                                                    nencodings_per_station = hyper[prefix]['nencodings_per_station'],
                                                    nencodings_per_lead_time = hyper[prefix]['nencodings_per_lead_time'],

                                                    number_of_features = hyper[prefix]['encoder_features'],
                                                    number_of_output_features = hyper[prefix]['encoder_output_features'],
                                                    number_of_encoders = hyper[prefix]['encoder_nattention_blocks'],
                                                    number_of_heads = hyper[prefix]['encoder_nheads'])

model_marginal: ModelMarginal = ModelMarginal(number_of_blocks = 1,
                                              number_of_knots  = 10,
                                              base_type = 0)

#model_marginal: ModelMarginal = ModelMarginal()

model_joint: ModelJoint = ModelJoint(incoming_features = hyper[prefix]['joint_incoming_features'],
                                     internal_features = hyper[prefix]['joint_internal_features'],
                                     number_of_outputs = 2 if censored else 1,
                                     input_vectors = 3 if censored else 2)

model: Model = Model(forecast_encoder = forecast_encoder, 
                     model_marginal = model_marginal,
                     model_joint = model_joint,
                     marginal_internal_features = hyper[prefix]['marginal_internal_features'],
                     censored = censored).to(device)

model_size: int = sum([p.numel() for p in model.parameters() if p.requires_grad])

print(f'Number of parameters: {model_size}')

training_stage: int = 0
model.freeze_joint()

optim = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

def loss_combined(x: torch.Tensor,
                  d: torch.Tensor,
                  s: torch.Tensor,
                  y: torch.Tensor,
                  j: torch.Tensor,
                  model: Model) -> torch.Tensor:

    embeddings: torch.Tensor = model.get_embeddings(x, d, s) 

    match training_stage:

        case 0:

            L_marginal = model.loss_marginal(embeddings, y[..., 0])

            return L_marginal.nanmean()

        case 1:
         
            t: torch.Tensor = torch.nn.functional.sigmoid(torch.randn((len(y), 1, 1), device = device)).clamp(max = 1 - 1e-12)
            y0_c: torch.Tensor = torch.randn((x.shape[0], y.shape[1], 1), dtype = torch.float32, device = x.device)

            v_c: torch.Tensor = y - y0_c
            yt_c: torch.Tensor = y0_c + t*v_c

            if censored:

                y0_d: torch.Tensor = (y0_c > 0.0).type(torch.float32) 
                y1_d: torch.Tensor = (y > 0.0).type(torch.float32)
                yt_d: torch.Tensor = torch.where(torch.rand_like(y0_d) < t, y1_d, y0_d)

                yt: torch.Tensor = torch.cat([yt_c, yt_d], dim = -1)

                L_reg, L_cls = model.loss_joint(embeddings, t, yt, v_c, y1_d)

                return L_reg.nanmean() + L_cls.nanmean()

            else:
                yt: torch.Tensor = yt_c
                y1_d: torch.Tensor = None

                L_reg = model.loss_joint(embeddings, t, yt, v_c, y1_d).nanmean()

                return L_reg.nanmean() 

        case _:
            raise RuntimeError('Invalid training stage.')


best_marginal: Model = None

def process_dataset(dataset: Dataset, 
                    loss_function: Callable) -> float:

    loss: float = 0.0
    counter: int = 0

    for i in tqdm(range(len(dataset))):

        j, s, d, _, x, y = dataset[i]

        if prefix == 'precipitation' and model.training:
            thr: torch.Tensor = (y.exp() - 1)*ysd.squeeze()*1000
            y = y.clone()
            y[thr >= 40] = torch.nan


        match training_stage:
        
            case 0:

                if y.isnan().all():
                    continue

            case 1:

                _idx: torch.Tensor = (~y.isnan()).all(dim = (-2, -1))

                s = s[_idx]
                d = d[_idx]
                x = x[_idx]
                y = y[_idx]

                if x.shape[0] == 0:
                    continue

        counter += 1

        loss_batch: torch.Tensor = loss_function(x, d, s, y, j, model)

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

    loss_train = process_dataset(dataset_train, loss_combined)
    model.eval()

    with torch.no_grad():
        loss_valid = process_dataset(dataset_valid, loss_combined)

    losses_train.append(loss_train)
    losses_valid.append(loss_valid)

    print(f"Train loss: {loss_train:.3f} Valid loss: {loss_valid:.3f}")
    print(f"Current best valid loss: {losses_valid[best_valid_loss_checkpoint]:.3f}")

    if loss_valid <= losses_valid[best_valid_loss_checkpoint]:
       
        model_path = join(output_path, "model")

        best_valid_loss_checkpoint = e
        best_valid_loss_model = deepcopy(model)

        if write_to_file:

            print(f'New best model found...saving and creating checkpoint...')

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
       
        match training_stage:

            case 0:

                print('End of case 0...')
                #torch.autograd.set_detect_anomaly(True)

                model = best_valid_loss_model
                best_marginal = deepcopy(model)
                best_valid_loss_checkpoint = e + 1

                model.freeze_encoder()
                model.freeze_marginal()
                model.freeze_joint(False)
                training_stage = 1
                tolerance = 20

                print(f'Switching to joint training {training_stage}...')

                optim = torch.optim.AdamW(model.model_joint.parameters(), lr = 1e-3, weight_decay = 1e-8)
                #base_optim = torch.optim.AdamW
                #optim = SAM(model.parameters(), base_optim, adaptive = True, lr = learning_rate, weight_decay = weight_decay)

            case 1:
                break

