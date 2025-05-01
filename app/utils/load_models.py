import torch
from tqdm.auto import tqdm
import wandb
import torch.nn.functional as F
import pdb
from utils.scheduler import *
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
from utils.model import Unet1D

### Model Parameters 


def load_model_and_weights(path_model):
    '''
    Loads the Unet1D model and its weights from the specified path.
    
    Args:
        path_model (str): The path to the model weights.
        
    Returns:
        model (Unet1D): The model with loaded weights.
    '''
    model = Unet1D(dim=8, dim_mults=(1, 2, 4, 8), channels=3)
    model.load_state_dict(torch.load(path_model, map_location=device))
    return model