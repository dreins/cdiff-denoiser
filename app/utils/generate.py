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
import app.utils.settings as s
from app.utils.load_models import load_model_and_weights


old_args = s.configure_args()


# Function to generate output (main logic from your generate function)
def generate(args, loader, noise_loader, index_to_trace_name, max_value_dict):
    device = torch.device(
        f"cuda:{old_args.gpu}" if torch.cuda.is_available() else "cpu")
    scheduler.initialize_parameters(old_args.T)
    model = load_model_and_weights(old_args.path_model).to(device)

    results = []
    T = old_args.T

    with torch.no_grad():
        model.eval()
        for eq_in, noise_in in tqdm(zip(loader, noise_loader), total=len(loader)):
            try:
                initial_eq_in = eq_in
                eq_in = eq_in[1].to(device)
                reduce_noise = random.randint(*old_args.Range_RNF) * 0.01
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
                    print(max_value_dict)
                    max_value = max_value_dict.get(
                        trace_name, 1.0)  # Default to 1 if not found
                    print(max_value)
                    print(restored_sampling_batch[idx][0])
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
