import argparse
import os

def configure_args():
    
    parser = argparse.ArgumentParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # GPU
    parser.add_argument("--T",
                        default=300,
                        type=int,
                        help="Timestep for each iteration (default: 300)")
    
    # GPU
    parser.add_argument("--gpu",
                        default=0,
                        type=int,
                        help="GPU index to use (default: 0)")
    # RNF
    parser.add_argument("--Range_RNF",
                        default=(40, 65),
                        type=tuple,
                        help="Range RNF as a tuple of two integers (min, max). Example: --Range_RNF 10, 20")

    parser.add_argument("--path_model",
                        default="~/cdiff-denoiser/models/model_best_3",
                        type=str,
                        help="Path to the model weights (default: full path)")

    # Parse arguments
    args = parser.parse_args()

    return args