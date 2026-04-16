import torch

def get_device():
    """Dynamically selects the best available hardware accelerator."""
    if torch.cuda.is_available():
        return torch.device("cuda") # For Windows/Linux NVIDIA GPUs
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # For macOS Apple Silicon
    else:
        return torch.device("cpu")  # Fallback