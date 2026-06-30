import os
import time
import glob
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from orchestrator import Orchestrator

def load_calibration_frames(cal_dir, filter_band="g"):
    print("Loading calibration frames...")
    try:
        bias_path = os.path.join(cal_dir, "median_bias.fts")
        with fits.open(bias_path) as hdul:
            bias = hdul[0].data.astype(float)
            
        flat_path = os.path.join(cal_dir, f"master_flat_{filter_band}.fit")
        with fits.open(flat_path) as hdul:
            flat = hdul[0].data.astype(float)
            
        # We don't have a master dark, assume 0 for now
        dark = np.zeros_like(bias)
        
        print("Calibration frames loaded successfully.")
        return bias, dark, flat
    except Exception as e:
        print(f"Failed to load calibration frames: {e}")
        return None, None, None

def plot_light_curves(photometry_engine, output_dir, frame_idx):
    """Plot the light curves for the first few tracked stars"""
    if not photometry_engine.light_curves:
        return
        
    plt.figure(figsize=(10, 4))
    # We plot a sample of 20 stars to keep the graph readable
    for star_id in list(photometry_engine.light_curves.keys())[:20]:
        plt.plot(photometry_engine.light_curves[star_id], label=f"Star {star_id}" if star_id < 5 else "")
    
    plt.title(f"Light Curves (Frame {frame_idx})")
    plt.xlabel("Frame Sequence")
    plt.ylabel("Flux")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"lightcurves_frame_{frame_idx}.png"))
    plt.close()

def main():
    data_dir = r"S:\Jean\Interns\Vihaan\20260509"
    cal_dir = r"S:\Jean\Interns\Vihaan\calibration frames"
    output_dir = "pipeline_test_output"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Setup Configuration
    target = "KIC11674677"
    filter_band = "g"
    
    # Load calibration frames
    bias, dark, flat = load_calibration_frames(cal_dir, filter_band)
    
    # Initialize the Orchestrator with real calibration frames
    print("Initializing Orchestrator...")
    orchestrator = Orchestrator(spool_directory=output_dir, dark=dark, flat=flat, bias=bias)
    
    # Override alert logger output directory so we can see outputs here
    orchestrator.alert_logger.output_dir = os.path.join(output_dir, "alerts")
    if not os.path.exists(orchestrator.alert_logger.output_dir):
         os.makedirs(orchestrator.alert_logger.output_dir)
    
    # 2. Get the correct sequence of files
    search_pattern = os.path.join(data_dir, f"*{target}*{filter_band}.fit*")
    fits_files = sorted(glob.glob(search_pattern))
    
    if not fits_files:
        print(f"No files found matching {search_pattern}")
        return
        
    print(f"Found {len(fits_files)} frames for target {target} in filter {filter_band}.")
    
    # 3. Emulate the Watchdog (feeding frames with delay)
    for idx, filepath in enumerate(fits_files[:10]):
        print(f"\n--- Simulating New Frame {idx+1}/{len(fits_files)} ---")
        orchestrator.process_new_image(filepath)
        
        # Visualize Light curves once we're past burn-in
        if orchestrator.state == "MONITORING":
            plot_light_curves(orchestrator.photometry_engine, output_dir, idx+1)
            
        print("Waiting 10 seconds for next frame...")
        # For our automated quick local test, we'll sleep for 0.1s. 
        # When you run this manually, change this to 10 or however long you want.
        time.sleep(0.1)

if __name__ == "__main__":
    main()
