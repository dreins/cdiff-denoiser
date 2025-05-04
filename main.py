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
import uvicorn

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

if __name__ == "__main__":
    T = int(getenv("T", 300))
    gpu = int(getenv("GPU", default=0))
    path_model = getenv("PATH_MODEL", default="~/cdiff-denoiser/models/model_best_4.pt")

    # Simulating inference or training process with CLI args
    # (Can be adapted to include real logic)
    print(f"[CLI] Running with T={T}, GPU={gpu}, model={path_model}")
    host = getenv("HOST", "0.0.0.0")
    port = int(getenv("PORT", 53053))
    uvicorn.run("main:app", host=host, port=port, reload=False)
