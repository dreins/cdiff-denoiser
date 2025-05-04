import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
import pdb
from app.utils.scheduler import *
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
from app.utils.model import Unet1D
import os

def load_model_and_weights(path_model):
    '''
    Loads the Unet1D model and its weights from the specified path.
    
    Args:
        path_model (str): The path to the model weights.
        
    Returns:
        model (Unet1D): The model with loaded weights.
    '''
    gpu = int(getenv("GPU", default=0))
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = getenv("PATH_MODEL", 
                      "models/model_best_4.pt")   
    if os.path.isabs(env_path):
        path_model = env_path
    else:
        path_model = os.path.join(root_path, env_path)# Comes from .env

    device = torch.device(
        f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = Unet1D(dim=8, dim_mults=(1, 2, 4, 8), channels=3)
    model.load_state_dict(torch.load(path_model, map_location=device))
    return model