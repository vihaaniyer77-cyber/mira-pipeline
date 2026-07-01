import os
import glob
import numpy as np
from astropy.io import fits

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from orchestrator import Orchestrator

def main():
    data_dir = r"S:\Jean\Interns\Vihaan\20260509"
    cal_dir = r"S:\Jean\Interns\Vihaan\calibration frames"
    target = "KIC8508931"
    filter_band = "i"
    
    # Quick calibration load
    with fits.open(os.path.join(cal_dir, "median_bias.fts")) as hdul:
        bias = hdul[0].data.astype(float)
    flat_path = os.path.join(cal_dir, f"master_flat_{filter_band}.fit")
    if not os.path.exists(flat_path):
        flat_path = os.path.join(cal_dir, f"mean_flat_{filter_band}.fts")
    with fits.open(flat_path) as hdul:
        flat = hdul[0].data.astype(float)
    dark = np.zeros_like(bias)
    
    orchestrator = Orchestrator(spool_directory=r"S:\Jean\Interns\Vihaan\pipeline_test_output", dark=dark, flat=flat, bias=bias)
    
    fits_files = sorted(glob.glob(os.path.join(data_dir, f"*{target}*{filter_band}.fit*")))[:20]
    print(f"Running optimization on first 20 frames of {target}...")
    
    # Monkey-patch alert logger so we don't spam output
    orchestrator.alert_logger.log_alert = lambda *args, **kwargs: None
    
    for filepath in fits_files:
        orchestrator.process_new_image(filepath)
        
    print("\n--- OPTIMIZATION RESULTS ---")
    light_curves = orchestrator.photometry_engine.light_curves
    
    max_fractional_std = 0.0
    max_z_score_pure_poisson = 0.0
    
    for star_id, fluxes in light_curves.items():
        if len(fluxes) > 10:
            mean = np.mean(fluxes[-10:])
            std = np.std(fluxes[-10:])
            if mean > 1000: # Only look at bright stable stars for fractional std
                frac_std = std / mean
                if frac_std > max_fractional_std:
                    max_fractional_std = frac_std
            
            # Pure poisson Z-score
            expected_noise = max(np.sqrt(abs(mean)), orchestrator.photometry_engine.min_std)
            z = (fluxes[-1] - mean) / (max(std, expected_noise) if max(std, expected_noise) > 0 else 1e-10)
            if abs(z) > max_z_score_pure_poisson:
                max_z_score_pure_poisson = abs(z)
                
    print(f"Maximum Natural Fractional Std (Atmospheric Wobble): {max_fractional_std*100:.3f}%")
    print(f"Maximum Natural Z-Score (Alignment Jitter limit): {max_z_score_pure_poisson:.2f}")
    
    # Recommend thresholds safely above the maximum natural noise floor
    recommended_fraction = max_fractional_std + 0.005 # pad by 0.5%
    recommended_z = np.ceil(max_z_score_pure_poisson + 1.0)
    
    print(f"\nRECOMMENDED ERROR FLOOR: {recommended_fraction*100:.2f}%")
    print(f"RECOMMENDED Z-SCORE THRESHOLD: {recommended_z:.1f}")

if __name__ == "__main__":
    main()
