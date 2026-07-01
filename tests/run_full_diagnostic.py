import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from orchestrator import Orchestrator
from calibration import align_image
from subtraction import optimal_image_subtraction

def load_calibration_frames(cal_dir, filter_band="i"):
    print("Loading calibration frames...")
    try:
        bias_path = os.path.join(cal_dir, "median_bias.fts")
        with fits.open(bias_path) as hdul:
            bias = hdul[0].data.astype(float)
            
        flat_path = os.path.join(cal_dir, f"master_flat_{filter_band}.fit")
        if not os.path.exists(flat_path):
            flat_path = os.path.join(cal_dir, f"mean_flat_{filter_band}.fts")
        with fits.open(flat_path) as hdul:
            flat = hdul[0].data.astype(float)
            
        dark = np.zeros_like(bias)
        print("Calibration frames loaded successfully.")
        return bias, dark, flat
    except Exception as e:
        print(f"Failed to load calibration frames: {e}")
        return None, None, None

def generate_aperture_qa_plot(image, tracked_stars_xy, output_dir):
    print("Generating Photometric Aperture QA plot...")
    plt.figure(figsize=(10, 10))
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(image)
    
    plt.imshow(image, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    plt.title(f"Aperture Overlay: {len(tracked_stars_xy)} Tracked Stars")
    
    for (x, y) in tracked_stars_xy:
        plt.plot(x, y, 'ro', markersize=10, fillstyle='none', markeredgewidth=1.0, alpha=0.5)
        
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "photometric_apertures.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved aperture overlay to {plot_path}")

def plot_anomaly_light_curves(photometry_engine, output_dir, frame_idx, anomalous_ids):
    if not anomalous_ids:
        return
        
    plt.figure(figsize=(10, 4))
    for star_id in sorted(list(anomalous_ids))[:20]:
        if star_id in photometry_engine.light_curves:
            plt.plot(photometry_engine.light_curves[star_id], label=f"Anomaly {star_id}")
    
    plt.title(f"Light Curves of Anomalous Stars (Frame {frame_idx})")
    plt.xlabel("Frame Sequence")
    plt.ylabel("Flux")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"anomaly_lightcurves.png"))
    plt.close()

def main():
    data_dir = r"S:\Jean\Interns\Vihaan\20260509"
    cal_dir = r"S:\Jean\Interns\Vihaan\calibration frames"
    output_dir = r"S:\Jean\Interns\Vihaan\test_pipeline_2"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    target = "KIC8508931"
    filter_band = "i"
    
    bias, dark, flat = load_calibration_frames(cal_dir, filter_band)
    
    print("Initializing Orchestrator...")
    orchestrator = Orchestrator(spool_directory=output_dir, dark=dark, flat=flat, bias=bias)
    
    orchestrator.alert_logger.output_dir = os.path.join(output_dir, "alerts")
    if not os.path.exists(orchestrator.alert_logger.output_dir):
         os.makedirs(orchestrator.alert_logger.output_dir)
    
    search_pattern = os.path.join(data_dir, f"*{target}*{filter_band}.fit*")
    fits_files = sorted(glob.glob(search_pattern))
    
    if not fits_files:
        print(f"No files found matching {search_pattern}")
        return
        
    print(f"Found {len(fits_files)} frames for target {target} in filter {filter_band}.")
    
    anomalous_stars = set()
    qa_plot_generated = False
    
    for idx, filepath in enumerate(fits_files):
        print(f"\n--- Processing Frame {idx+1}/{len(fits_files)} ---")
        orchestrator.process_new_image(filepath)
        
        if orchestrator.state == "MONITORING":
            if not qa_plot_generated:
                generate_aperture_qa_plot(orchestrator.reference_image, orchestrator.background_stars_xy, output_dir)
                qa_plot_generated = True

            for i, history in orchestrator.photometry_engine.light_curves.items():
                if len(history) > 2:
                    mean_flux = np.mean(history[-10:])
                    std = np.std(history[-10:])
                    expected = max(np.sqrt(abs(mean_flux)), orchestrator.photometry_engine.min_std)
                    if std > expected * 3.0:
                        anomalous_stars.add(i)
                    z = (history[-1] - mean_flux) / (max(std, expected) if max(std, expected) > 0 else 1e-10)
                    if abs(z) > 4.0:
                        anomalous_stars.add(i)
                        
            if idx % 5 == 0:
                plot_anomaly_light_curves(orchestrator.photometry_engine, output_dir, idx+1, anomalous_stars)
            
            if idx % 10 == 0:
                print("Generating manual difference image...")
                with fits.open(filepath) as hdul:
                    raw_image = hdul[0].data.astype(float)
                try:
                    clean_img = (raw_image - bias - dark) / (flat / np.median(flat))
                    aligned = align_image(clean_img, orchestrator.reference_image)
                    diff = optimal_image_subtraction(aligned, orchestrator.reference_image)
                    
                    plt.figure(figsize=(8,8))
                    zscale = ZScaleInterval()
                    vmin, vmax = zscale.get_limits(diff)
                    plt.imshow(diff, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
                    plt.title(f"Difference Image (Target - Ref) Frame {idx+1}")
                    diff_path = os.path.join(output_dir, f"difference_frame_{idx+1}.png")
                    plt.savefig(diff_path)
                    plt.close()
                except Exception as e:
                    print(f"Skipping diff image due to error: {e}")

if __name__ == "__main__":
    main()
