import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
import pdb
from app.utils.scheduler import *
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
import app.utils.scheduler as scheduler
import os
import tensorflow as tf
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from app.utils.load_models import load_model_and_weights
from os import getenv
from dotenv import load_dotenv

load_dotenv()

# Function to generate output (main logic from your generate function)
def generate(loader, noise_loader, index_to_trace_name, max_value_dict):
    T = int(getenv("T", 300))
    gpu = int(getenv("GPU", default=0))
    path_model = getenv("PATH_MODEL", default="~/cdiff-denoiser/models/model_best_4.pt")
    Range_RNF = getenv("RANGE_RNF", default=(40, 65))
    Range_RNF = tuple(map(int, Range_RNF.split(",")))

    device = torch.device(
        f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    scheduler.initialize_parameters(T)
    model = load_model_and_weights(path_model).to(device)

    results = []

    with torch.no_grad():
        model.eval()
        for eq_in, noise_in in tqdm(zip(loader, noise_loader), total=len(loader)):
            try:
                initial_eq_in = eq_in
                eq_in = eq_in[1].to(device)
                reduce_noise = random.randint(Range_RNF[0], Range_RNF[1]) * 0.01
                noise_real = noise_in[1].to(device) * reduce_noise
                signal_noisy = eq_in + noise_real
                t = torch.tensor([T - 1]).long().to(device)
                t = T-1
                # Denoising
                restored_sample = scheduler.sample(
                    model, signal_noisy.float(), t, batch_size=signal_noisy.shape[0])
                restored_sampling_batch = [x.cpu().numpy()
                                           for x in restored_sample[-1]]

                # Get trace names
                trace_names = [index_to_trace_name[idx.item()]
                               for idx in initial_eq_in[0]]
                
                print(trace_names)

                # Process each sample
                for idx, trace_name in enumerate(trace_names):
                    # Multiply results by the max values
                    max_value = max_value_dict.get(
                        trace_name, 1.0)  # Default to 1 if not found
                    results.append({
                        "trace_name": trace_name,
                        "E_channel_denoised": (restored_sampling_batch[idx][0] * max_value).tolist(),
                        "N_channel_denoised": (restored_sampling_batch[idx][1] * max_value).tolist(),
                        "Z_channel_denoised": (restored_sampling_batch[idx][2] * max_value).tolist()
                    })
            except KeyError as e:
                print(f"[Skipped] Missing index in index_to_trace_name: {e}")
                continue

    return results