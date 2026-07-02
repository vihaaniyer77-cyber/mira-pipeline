import os
import sys
import shutil
import time
import numpy as np
from astropy.io import fits
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from orchestrator import Orchestrator
import calibration
import astrometry_solver

# --- Monkeypatching ---
def mock_align_image(target_image, reference_image):
    return target_image
calibration.align_image = mock_align_image

def mock_solve_wcs(filepath):
    return None
astrometry_solver.solve_wcs_for_image = mock_solve_wcs

# --- Configuration ---
IMG_SIZE = 2048
NUM_FRAMES = 100
BURN_IN_FRAMES = 3
TOTAL_FRAMES = NUM_FRAMES + BURN_IN_FRAMES

NUM_FLAT = 800
NUM_SLIGHT_FLARE = 175
NUM_OBVIOUS_FLARE = 175
NUM_SLIGHT_PULSE = 175
NUM_OBVIOUS_PULSE = 175

TOTAL_STARS = NUM_FLAT + NUM_SLIGHT_FLARE + NUM_OBVIOUS_FLARE + NUM_SLIGHT_PULSE + NUM_OBVIOUS_PULSE

SPOOL_DIR = os.path.join(os.path.dirname(__file__), "simulated_spool")
OUT_DIR = os.path.join(os.path.dirname(__file__), "simulated_output")

# Generate coordinates safely away from the edge (margin of 20)
np.random.seed(42)
all_xs = np.random.randint(20, IMG_SIZE - 20, size=TOTAL_STARS)
all_ys = np.random.randint(20, IMG_SIZE - 20, size=TOTAL_STARS)

# Baseline fluxes (between 500 and 15000)
base_fluxes = np.random.uniform(500, 15000, size=TOTAL_STARS)

# Categorize
flat_idx = np.arange(0, NUM_FLAT)
sflare_idx = np.arange(NUM_FLAT, NUM_FLAT + NUM_SLIGHT_FLARE)
oflare_idx = np.arange(sflare_idx[-1]+1, sflare_idx[-1]+1 + NUM_OBVIOUS_FLARE)
spulse_idx = np.arange(oflare_idx[-1]+1, oflare_idx[-1]+1 + NUM_SLIGHT_PULSE)
opulse_idx = np.arange(spulse_idx[-1]+1, spulse_idx[-1]+1 + NUM_OBVIOUS_PULSE)

def create_gaussian_psf(size=11, fwhm=3.0):
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    sigma = fwhm / 2.3548
    return np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))

def generate_simulated_data():
    if os.path.exists(SPOOL_DIR):
        shutil.rmtree(SPOOL_DIR)
    os.makedirs(SPOOL_DIR)
    
    psf = create_gaussian_psf(size=11, fwhm=3.0)
    
    print("Generating simulated frames...")
    # Generate flare triggers at specific random frames (between frame 20 and 80)
    flare_frames = np.random.randint(20, 80, size=TOTAL_STARS)
    
    for frame_i in range(TOTAL_FRAMES):
        img = np.random.normal(loc=200, scale=10, size=(IMG_SIZE, IMG_SIZE)).astype(np.float32)
        
        for i in range(TOTAL_STARS):
            x, y = all_xs[i], all_ys[i]
            flux = base_fluxes[i]
            
            if i in sflare_idx:
                if frame_i == flare_frames[i]:
                    flux *= np.random.uniform(1.5, 2.0)
            elif i in oflare_idx:
                if frame_i == flare_frames[i]:
                    flux *= np.random.uniform(5.0, 10.0)
            elif i in spulse_idx:
                # Slight pulse +/- 15%
                flux *= 1.0 + 0.15 * np.sin(2 * np.pi * frame_i / 30.0)
            elif i in opulse_idx:
                # Obvious pulse +/- 60%
                flux *= 1.0 + 0.60 * np.sin(2 * np.pi * frame_i / 30.0)
                
            # Add star to image
            psf_scaled = psf * flux
            # Add Poisson noise to the star
            psf_noisy = np.random.poisson(np.clip(psf_scaled, 0, None))
            
            sy, sx = psf_noisy.shape
            img[y - sy//2 : y - sy//2 + sy, x - sx//2 : x - sx//2 + sx] += psf_noisy
            
        img = np.clip(img, 0, 65535).astype(np.uint16)
        
        fits_filename = os.path.join(SPOOL_DIR, f"sim_{frame_i:04d}.fits")
        hdu = fits.PrimaryHDU(img)
        hdu.writeto(fits_filename, overwrite=True)

def run_simulation():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)
    
    orchestrator = Orchestrator(spool_directory=SPOOL_DIR)
    orchestrator.bad_pixel_mask = None
    
    # Mock log_alert to skip PNG generation for speed
    def mock_log_alert(engine_name, x, y, full_image, crop_size=50, wcs=None):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        engine_prefix = "A" if "A" in engine_name else "B"
        txt_path = os.path.join(orchestrator.alert_logger.log_dir, f"{timestamp}_{engine_prefix}_X{int(x)}_Y{int(y)}.txt")
        with open(txt_path, "w") as f:
            f.write(f"Engine: {engine_name}\n")
            f.write(f"X: {x}, Y: {y}\n")
    orchestrator.alert_logger.log_alert = mock_log_alert
    # Redirect output dir to our test dir so we don't mess up main
    orchestrator.alert_logger.log_dir = os.path.join(OUT_DIR, "alerts")
    if not os.path.exists(orchestrator.alert_logger.log_dir):
        os.makedirs(orchestrator.alert_logger.log_dir)
        
    start_time = time.time()
    
    # Process sequentially
    frame_files = sorted(os.listdir(SPOOL_DIR))
    for f in frame_files:
        filepath = os.path.join(SPOOL_DIR, f)
        orchestrator.process_new_image(filepath)
        
    end_time = time.time()
    
    # Metrics
    total_time = end_time - start_time
    time_per_frame = total_time / len(frame_files)
    
    print("\n" + "="*50)
    print("SIMULATION RESULTS")
    print("="*50)
    print(f"Total Time: {total_time:.2f} s")
    print(f"Time per frame: {time_per_frame:.4f} s")
    print(f"Frames processed: {len(frame_files)}")
    
    # Parse alerts
    alerts_found = []
    if os.path.exists(orchestrator.alert_logger.log_dir):
        for f in os.listdir(orchestrator.alert_logger.log_dir):
            if f.endswith('.txt'):
                parts = f.split('_')
                if "X" in parts[-2] and "Y" in parts[-1]:
                    x = float(parts[-2][1:])
                    y = float(parts[-1][1:].replace('.txt', ''))
                    alerts_found.append((x, y))
                    
    # Evaluate
    # Any alert within 2.0 pixels of a true star counts as a detection for that star
    def count_hits(target_indices):
        hits = 0
        misses = 0
        for idx in target_indices:
            tx, ty = all_xs[idx], all_ys[idx]
            detected = False
            for ax, ay in alerts_found:
                dist = np.sqrt((ax - tx)**2 + (ay - ty)**2)
                if dist < 3.0:
                    detected = True
                    break
            if detected:
                hits += 1
            else:
                misses += 1
        return hits, misses

    sflare_h, sflare_m = count_hits(sflare_idx)
    oflare_h, oflare_m = count_hits(oflare_idx)
    spulse_h, spulse_m = count_hits(spulse_idx)
    opulse_h, opulse_m = count_hits(opulse_idx)
    
    # Flat stars should have 0 hits (these are false positives)
    fp_hits, _ = count_hits(flat_idx)
    
    print("\n--- Detection Metrics ---")
    print(f"Slight Flares:     {sflare_h}/{NUM_SLIGHT_FLARE} Detected ({sflare_m} Missed)")
    print(f"Obvious Flares:    {oflare_h}/{NUM_OBVIOUS_FLARE} Detected ({oflare_m} Missed)")
    print(f"Slight Pulsators:  {spulse_h}/{NUM_SLIGHT_PULSE} Detected ({spulse_m} Missed)")
    print(f"Obvious Pulsators: {opulse_h}/{NUM_OBVIOUS_PULSE} Detected ({opulse_m} Missed)")
    
    print(f"\nFalse Positives (from {NUM_FLAT} flat stars): {fp_hits}")
    print("="*50)

if __name__ == "__main__":
    generate_simulated_data()
    run_simulation()
