from pydantic import BaseModel
from typing import List, Dict, Optional


# === INPUT STRUCTURE FOR INFERENCE ===
class InputSeismicTrace(BaseModel):
    trace_name: str
    Z_channel: List[float]
    E_channel: List[float]
    N_channel: List[float]


class InputDataWaveform(BaseModel):
    data: List[InputSeismicTrace]


# === OUTPUT STRUCTURE FOR INFERENCE ===
class OutputTraceResult(BaseModel):
    trace_name: str
    Z_channel_denoised: List[float]
    E_channel_denoised: List[float]
    N_channel_denoised: List[float]


class OutputDataWaveform(BaseModel):
    results: List[OutputTraceResult]
