import numpy as np

def apply_poisson_gaussian_noise(image_array, k=100.0, sigma=0.03):
    """Simulates shot noise (Poisson) and thermal amplifier gain (Gaussian)."""
    scaled = np.clip(image_array, 0.0, 1.0) * k
    poisson_noisy = np.random.poisson(scaled) / k
    gaussian_noise = np.random.normal(0.0, sigma, image_array.shape)
    return np.clip(poisson_noisy + gaussian_noise, 0.0, 1.0).astype(np.float32)

def apply_tone_curve_compression(image_array, gamma=1.5):
    """Simulates ISP dynamic range compression pushing extreme values into bounds."""
    return np.clip(np.power(np.clip(image_array, 0.0, 1.0), gamma), 0.0, 1.0).astype(np.float32)

def apply_chromatic_adaptation(image_array, gains=(1.1, 0.95, 0.9)):
    """Simulates illuminant shift via von Kries diagonal scaling."""
    result = image_array.copy()
    for c in range(3):
        result[:, :, c] *= gains[c]
    return np.clip(result, 0.0, 1.0).astype(np.float32)

def perturb_image(image_array, noise_type="poisson_gaussian", severity=0.03):
    if noise_type == "poisson_gaussian":
        return apply_poisson_gaussian_noise(image_array, sigma=severity)
    elif noise_type == "tone_curve":
        return apply_tone_curve_compression(image_array, gamma=1.0 + severity * 10)
    elif noise_type == "chromatic":
        return apply_chromatic_adaptation(image_array, gains=(1.0 + severity, 1.0, 1.0 - severity))
    return image_array
