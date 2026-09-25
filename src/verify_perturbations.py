import os
import numpy as np
from PIL import Image
from perturb import perturb_image

def verify_noise_engine():
    print("=" * 60)
    print("QADR Phase 2: Sensor Noise Engine Verification")
    print("=" * 60)
    
    os.makedirs("data/samples", exist_ok=True)
    
    # Generate synthetic RGB test pattern (224, 224, 3)
    dummy_img = np.random.rand(224, 224, 3).astype(np.float32)
    
    severities = [0.01, 0.03, 0.05, 0.10]
    for sig in severities:
        perturbed = perturb_image(dummy_img, noise_type="poisson_gaussian", severity=sig)
        
        # Assertions
        assert perturbed.shape == (224, 224, 3), f"Shape mismatch: {perturbed.shape}"
        assert perturbed.dtype == np.float32, f"Dtype mismatch: {perturbed.dtype}"
        assert not np.isnan(perturbed).any(), "NaN values detected!"
        assert not np.isinf(perturbed).any(), "Inf values detected!"
        assert np.min(perturbed) >= 0.0 and np.max(perturbed) <= 1.0, "Values out of bounds [0, 1]"
        
        # Save sample output
        save_arr = (perturbed * 255.0).astype(np.uint8)
        Image.fromarray(save_arr).save(f"data/samples/noise_sig_{sig:.2f}.jpg")
        print(f"[+] Verified Severity σ={sig:.2f} | Min: {np.min(perturbed):.3f} | Max: {np.max(perturbed):.3f} | PASS")
        
    print("=" * 60)
    print("[STATUS] Noise engine fully verified with 0 numerical anomalies.")
    print("=" * 60)

if __name__ == "__main__":
    verify_noise_engine()
