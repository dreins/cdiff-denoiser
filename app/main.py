# from fastapi import FastAPI
# from dotenv import load_dotenv
# from fastapi.middleware.cors import CORSMiddleware
# from app.utils.pydantic import *
# from app.utils.helper import create_inference_dataloader, generate_noise_dataframe, validate_trace_lengths
# import pandas as pd
# import app.utils.settings as s
# from app.utils.generate import generate
# import sys
# import os

# import logging

# logging.basicConfig(level=logging.INFO) 
# logger = logging.getLogger(__name__)


# load_dotenv()
# app = FastAPI()
# origins = ["*"]


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )



# @app.post("/inference")
# def inference(input_data: InputDataWaveform):
    
#     rows = [trace.dict() for trace in input_data.data]
#     df = pd.DataFrame(rows)
    
    
#     try:
#         validate_trace_lengths(df)
#     except ValueError as e:
#         print(str(e))
#         sys.exit(1)  # Stop execution
        
#     noise_first_row = df.iloc[0]
#     noise_channel_length = len(noise_first_row['Z_channel']) 
        
#     df_noise = generate_noise_dataframe(num_traces=len(rows), length=noise_channel_length, noise_std=0.05)

#     # Create dataloader
#     loader, index_to_trace_name, max_values = create_inference_dataloader(df)
#     noise_loader, _, _ = create_inference_dataloader(df_noise)

#     # Run the generate function
#     old_args = s.configure_args()
#     results = generate(old_args, loader, noise_loader, index_to_trace_name, max_values)

#     # Convert results to OutputTraceResult format
#     output_results = [
#         OutputTraceResult(
#             trace_name=result["trace_name"],
#             E_channel_synthetic=result["restored_sampling_E"],  # Ensure correct field name
#             N_channel_synthetic=result["restored_sampling_N"],  # Ensure correct field name
#             Z_channel_synthetic=result["restored_sampling_Z"]   # Ensure correct field name
#         )
#         for result in results
#     ]

#     output = OutputDataWaveform(results=output_results)
#     return output


# if __name__ == "__main__":
#     config = uvicorn.Config(
#         "main:app", port=53053, log_level="info", host="0.0.0.0"
#     )
#     server = uvicorn.Server(config)
#     server.run()

import argparse
import os
from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from app.utils.pydantic import *
from app.utils.helper import create_inference_dataloader, generate_noise_dataframe, validate_trace_lengths
import pandas as pd
from app.utils.generate import generate
import sys
import logging
from os import getenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
async def inference(input_data: InputDataWaveform):
    rows = [trace.dict() for trace in input_data.data]
    df = pd.DataFrame(rows)
    
    try:
        validate_trace_lengths(df)
    except ValueError as e:
        print(str(e))
        sys.exit(1)
        
    noise_first_row = df.iloc[0]
    noise_channel_length = len(noise_first_row['Z_channel']) 
        
    df_noise = generate_noise_dataframe(num_traces=len(rows), length=noise_channel_length)
    
    loader, index_to_trace_name, max_values = create_inference_dataloader(df)
    
    noise_loader, _, _ = create_inference_dataloader(df_noise)
    results = generate(loader, noise_loader, index_to_trace_name, max_values)

    output_results = [
        OutputTraceResult(
            trace_name=result["trace_name"],
            E_channel_denoised=result["E_channel_denoised"],
            N_channel_denoised=result["N_channel_denoised"],
            Z_channel_denoised=result["Z_channel_denoised"]
        )
        for result in results
    ]

    output = OutputDataWaveform(results=output_results)
    return output


# ----------------------------
# CLI MODE (optional training/inference outside API)
# ----------------------------

# def configure_args():
#     parser = argparse.ArgumentParser(description="Cold Diffusion Model Arguments")
#     current_dir = os.path.dirname(os.path.abspath(__file__))
    
#     parser.add_argument("--T",
#                         default=300,
#                         type=int,
#                         help="Timestep for each iteration (default: 300)")

#     parser.add_argument("--gpu",
#                         default=0,
#                         type=int,
#                         help="GPU index to use (default: 0)")

#     parser.add_argument("--Range_RNF",
#                         default=(40, 65),
#                         type=tuple,
#                         help="Range RNF as a tuple of two integers (min, max). Example: --Range_RNF 10, 20")

#     parser.add_argument("--path_model",
#                         default="~/cdiff-denoiser/models/model_best_4.pt",
#                         type=str,
#                         help="Path to the model weights (default: full path)")

#     args = parser.parse_args()
#     return args

def run_cli():
    args = configure_args()
    T = int(getenv("T", 300))
    gpu = int(getenv("GPU", default=0))
    path_model = getenv("PATH_MODEL", default="~/cdiff-denoiser/models/model_best_4.pt")
    Range_RNF = getenv("RANGE_RNF", default=(40, 65))

    # Simulating inference or training process with CLI args
    # (Can be adapted to include real logic)
    print(f"[CLI] Running with T={T}, GPU={gpu}, model={path_model}")

    # Implement logic based on the CLI arguments (if necessary)
    # e.g., load model and perform inference or training based on input data

if __name__ == "__main__":
    run_cli()
