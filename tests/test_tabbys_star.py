import os
import time
import glob
import numpy as np
from astropy.io import fits
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from orchestrator import Orchestrator

def load_calibration_frames(cal_dir, filter_band="i"):
    print("Loading calibration frames...")
    try:
        bias_path = os.path.join(cal_dir, "median_bias.fts")
        with fits.open(bias_path) as hdul:
            bias = hdul[0].data.astype(float)
            
        flat_path = os.path.join(cal_dir, f"master_flat_{filter_band}.fit")
        with fits.open(flat_path) as hdul:
            flat = hdul[0].data.astype(float)
            
        dark = np.zeros_like(bias)
        return bias, dark, flat
    except Exception as e:
        print(f"Failed to load calibration frames: {e}")
        return None, None, None

@patch('orchestrator.optimal_image_subtraction')
def main(mock_subtraction):
    # Mock optimal image subtraction to return a blank image to skip the 15 minute math phase
    mock_subtraction.side_effect = lambda tgt, ref: np.zeros_like(tgt)

    data_dir = r"S:\Jean\Interns\Vihaan\20260509"
    cal_dir = r"S:\Jean\Interns\Vihaan\calibration frames"
    output_dir = "tests/tabbys_star_output"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    target = "KIC8462852" # Tabby's Star (Known for massive erratic dips)
    filter_band = "i"
    
    bias, dark, flat = load_calibration_frames(cal_dir, filter_band)
    
    print("Initializing Orchestrator for Photometry Alert Test (Engine A Disabled)...")
    orchestrator = Orchestrator(spool_directory=output_dir, dark=dark, flat=flat, bias=bias)
    
    orchestrator.alert_logger.output_dir = os.path.join(output_dir, "alerts")
    if not os.path.exists(orchestrator.alert_logger.output_dir):
         os.makedirs(orchestrator.alert_logger.output_dir)
    
    search_pattern = os.path.join(data_dir, f"*{target}*{filter_band}.fit*")
    fits_files = sorted(glob.glob(search_pattern))
    
    if not fits_files:
        print(f"No files found matching {search_pattern}")
        return
        
    print(f"Found {len(fits_files)} frames for {target}. Running...")
    
    for idx, filepath in enumerate(fits_files):
        print(f"\n--- Frame {idx+1}/{len(fits_files)} ---")
        orchestrator.process_new_image(filepath)

if __name__ == "__main__":
    main()
