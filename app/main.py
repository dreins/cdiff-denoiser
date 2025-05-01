from fastapi import FastAPI, BackgroundTasks
from obspy import UTCDateTime
import uvicorn
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
from app.utils.pydantic import *
from app.utils.helper import create_inference_dataloader, generate_noise_dataframe, validate_trace_lengths
import pandas as pd
import numpy as np
import json
import h5py
import random
import tensorflow as tf
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Optional 
import app.utils.settings as s
from app.utils.generate import generate


old_args = s.configure_args()


load_dotenv()
app = FastAPI()
origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/inference")
def inference(input_data: InputDataWaveform):
    rows = [trace.dict() for trace in input_data.data]
    df = pd.DataFrame(rows)
    
    
    try:
        validate_trace_lengths(df)
    except ValueError as e:
        print(str(e))
        sys.exit(1)  # Stop execution
        
    noise_first_row = df.iloc[0]
    noise_channel_length = len(noise_first_row['Z_channel']) 
        
    df_noise = generate_noise_dataframe(num_traces=len(rows), length=noise_channel_length, noise_std=0.05)

    # Create dataloader
    loader, index_to_trace_name, max_values = create_inference_dataloader(df)
    noise_loader, _, _ = create_inference_dataloader(df_noise)

    # Run the generate function
    results = generate(old_args, loader, noise_loader, index_to_trace_name, max_values)

    # Convert results to OutputTraceResult format
    output_results = [
        OutputTraceResult(
            trace_name=result["trace_name"],
            E_channel_synthetic=result["restored_sampling_E"],  # Ensure correct field name
            N_channel_synthetic=result["restored_sampling_N"],  # Ensure correct field name
            Z_channel_synthetic=result["restored_sampling_Z"]   # Ensure correct field name
        )
        for result in results
    ]

    output = OutputDataWaveform(results=output_results)
    return output


# if __name__ == "__main__":
#     config = uvicorn.Config(
#         "main:app", port=53053, log_level="info", host="0.0.0.0"
#     )
#     server = uvicorn.Server(config)
#     server.run()