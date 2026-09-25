import numpy as np

def apply_qadr(tensor, alpha=1.0, beta=0.0, gamma_min=-2.5, gamma_max=2.5):
    """
    QADR (Quantization-Aware Dynamic Rescaling):
    x_scaled = alpha * clamp(x + beta, gamma_min, gamma_max)
    """
    clamped = np.clip(tensor + beta, gamma_min, gamma_max)
    scaled = alpha * clamped
    return scaled.astype(np.float32)
