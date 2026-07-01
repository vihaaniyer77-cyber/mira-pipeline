import os
import time
import glob
import numpy as np
from astropy.io import fits
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from orchestrator import Orchestrator

def load_calibration_frames(cal_dir, filter_band="g"):
    print("Loading calibration frames...")
    try:
        bias_path = os.path.join(cal_dir, "Bias-0001.fit")  # Found in 20260626
        with fits.open(bias_path) as hdul:
            bias = hdul[0].data.astype(float)
            
        flat_path = os.path.join(cal_dir, f"SkyFlat-0001{filter_band}.fit") # Found in 20260626
        with fits.open(flat_path) as hdul:
            flat = hdul[0].data.astype(float)
            
        dark = np.zeros_like(bias)
        
        print("Calibration frames loaded successfully.")
        return bias, dark, flat
    except Exception as e:
        print(f"Failed to load calibration frames: {e}")
        return None, None, None

def main():
    data_dir = r"S:\Jean\Interns\Vihaan\20260626"
    cal_dir = data_dir
    output_dir = "tests/comet_test_output"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    target = "Comet_J3"
    filter_band = "g"
    
    bias, dark, flat = load_calibration_frames(cal_dir, filter_band)
    
    print("Initializing Orchestrator for Discovery Engine Test...")
    orchestrator = Orchestrator(spool_directory=output_dir, dark=dark, flat=flat, bias=bias)
    
    orchestrator.alert_logger.output_dir = os.path.join(output_dir, "alerts")
    if not os.path.exists(orchestrator.alert_logger.output_dir):
         os.makedirs(orchestrator.alert_logger.output_dir)
    
    search_pattern = os.path.join(data_dir, f"*{target}*{filter_band}.fit*")
    fits_files = sorted(glob.glob(search_pattern))
    
    if not fits_files:
        print(f"No files found matching {search_pattern}")
        return
        
    print(f"Found {len(fits_files)} frames for {target}. Running full transient detection...")
    
    for idx, filepath in enumerate(fits_files):
        print(f"\n--- Frame {idx+1}/{len(fits_files)} ---")
        orchestrator.process_new_image(filepath)
        time.sleep(0.01)

if __name__ == "__main__":
    main()
