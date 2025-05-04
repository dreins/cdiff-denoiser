import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from os import getenv


load_dotenv()

def cosine_beta_schedule(timesteps, s=0.008):
    """Creates a cosine beta schedule."""
    beta_start = 0.0001
    beta_end = 0.02
    steps = torch.arange(timesteps)
    beta_t = torch.cos(((steps / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    beta_t = (
        beta_t * (beta_end - beta_start) + beta_start
    )  # Scale the beta values according to the specification
    return beta_t


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_start, x_end, t):
    return get_index_from_list(
        sqrt_alphas_cumprod, t, x_start.shape
    ) * x_start + get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    ) * (
        x_end
    )


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

T = None
betas = None
alphas = None
alphas_cumprod = None
alphas_cumprod_prev = None
sqrt_recip_alphas = None
sqrt_alphas_cumprod = None
sqrt_one_minus_alphas_cumprod = None
posterior_variance = None


def initialize_parameters(timesteps):
    global device, T, betas, alphas, alphas_cumprod, alphas_cumprod_prev, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance
    T = int(getenv("T", 300))
    gpu = int(getenv("GPU", default=0))
    path_model = getenv("PATH_MODEL", default="~/cdiff-denoiser/models/model_best_4.pt")
    Range_RNF = getenv("RANGE_RNF", default=(40, 65))
    
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    betas = cosine_beta_schedule(timesteps=T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

@torch.no_grad()
def direct_denoising(model, x, t):
    model.eval()
    model_mean = model(x, t)
    return model_mean


@torch.no_grad()
def sample(model, img, t, batch_size=4):

    model.eval()

    while t:
        xt = img
        step = torch.full((batch_size,), t - 1, dtype=torch.long).to(device)

        x1_bar = model(img, step) 
        x2_bar = get_x2_bar_from_xt(x1_bar, img, step) 

        xt_bar = x1_bar
        if t != 0:
            xt_bar = forward_diffusion_sample(x_start=xt_bar, x_end=x2_bar, t=step)

        xt_sub1_bar = x1_bar

        # Questa è la parte vitale dove riaggiung D(x,s-1)
        if t - 1 != 0:
            step2 = torch.full((batch_size,), t - 2, dtype=torch.long).to(device)

            xt_sub1_bar = forward_diffusion_sample(
                x_start=xt_sub1_bar, x_end=x2_bar, t=step2
            )

        x = img - xt_bar + xt_sub1_bar
        img = x
        t = t - 1

    return xt, img

def get_x2_bar_from_xt(x1_bar, xt, t):
    return (
        xt
        - (get_index_from_list(sqrt_alphas_cumprod, t, x1_bar.shape) * x1_bar)
        - (get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x1_bar.shape) * x1_bar)
    ) / get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x1_bar.shape)