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
import orchestrator
import calibration
import astrometry_solver

# --- Monkeypatching ---
def mock_align_image(target_image, reference_image):
    return target_image
# Properly patch it inside orchestrator so it actually bypasses it!
orchestrator.align_image = mock_align_image

def mock_solve_wcs(filepath):
    return None
astrometry_solver.solve_wcs_for_image = mock_solve_wcs

# --- Configuration ---
IMG_SIZE = 2048
NUM_FRAMES = 100
BURN_IN_FRAMES = 3
TOTAL_FRAMES = NUM_FRAMES + BURN_IN_FRAMES

# The 8:7 Ratio (800 Flat, 700 Variables)
NUM_FLAT = 800
NUM_SLIGHT_FLARE = 140
NUM_OBVIOUS_FLARE = 140
NUM_SLIGHT_PULSE = 140
NUM_OBVIOUS_PULSE = 140
NUM_SUPERNOVA = 140

TOTAL_STARS = NUM_FLAT + NUM_SLIGHT_FLARE + NUM_OBVIOUS_FLARE + NUM_SLIGHT_PULSE + NUM_OBVIOUS_PULSE + NUM_SUPERNOVA
NUM_HOT_PIXELS = 6000

SPOOL_DIR = os.path.join(os.path.dirname(__file__), "simulated_spool")
OUT_DIR = os.path.join(os.path.dirname(__file__), "simulated_output")
HOT_PIXEL_MOCK_PATH = os.path.join(os.path.dirname(__file__), "mock_hot_pixels.fts")

# Generate coordinates safely away from the edge (margin of 20)
np.random.seed(42)
all_xs = np.random.randint(20, IMG_SIZE - 20, size=TOTAL_STARS)
all_ys = np.random.randint(20, IMG_SIZE - 20, size=TOTAL_STARS)

# Generate hot pixel coordinates
hot_xs = np.random.randint(0, IMG_SIZE, size=NUM_HOT_PIXELS)
hot_ys = np.random.randint(0, IMG_SIZE, size=NUM_HOT_PIXELS)
hot_fluxes = np.random.uniform(1000, 20000, size=NUM_HOT_PIXELS)

# Baseline fluxes (between 500 and 15000)
base_fluxes = np.random.uniform(500, 15000, size=TOTAL_STARS)

# Categorize
flat_idx = np.arange(0, NUM_FLAT)
sflare_idx = np.arange(flat_idx[-1]+1, flat_idx[-1]+1 + NUM_SLIGHT_FLARE)
oflare_idx = np.arange(sflare_idx[-1]+1, sflare_idx[-1]+1 + NUM_OBVIOUS_FLARE)
spulse_idx = np.arange(oflare_idx[-1]+1, oflare_idx[-1]+1 + NUM_SLIGHT_PULSE)
opulse_idx = np.arange(spulse_idx[-1]+1, spulse_idx[-1]+1 + NUM_OBVIOUS_PULSE)
sn_idx = np.arange(opulse_idx[-1]+1, opulse_idx[-1]+1 + NUM_SUPERNOVA)

def create_gaussian_psf(size=11, fwhm=3.0):
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    sigma = fwhm / 2.3548
    return np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))

def generate_hot_pixel_file():
    print("Generating mock hot pixel master dark...")
    dark_data = np.random.normal(loc=500, scale=30, size=(IMG_SIZE, IMG_SIZE)).astype(np.float32)
    # Inject the hot pixels
    dark_data[hot_ys, hot_xs] = hot_fluxes
    hdu = fits.PrimaryHDU(dark_data)
    hdu.writeto(HOT_PIXEL_MOCK_PATH, overwrite=True)

def generate_simulated_data():
    if os.path.exists(SPOOL_DIR):
        shutil.rmtree(SPOOL_DIR)
    os.makedirs(SPOOL_DIR)
    
    psf = create_gaussian_psf(size=11, fwhm=3.0)
    
    print("Generating simulated frames...")
    # Generate event triggers at specific random frames (between frame 20 and 80)
    event_frames = np.random.randint(20, 80, size=TOTAL_STARS)
    
    for frame_i in range(TOTAL_FRAMES):
        # Progress bar
        sys.stdout.write(f"\rGenerating Data... [{frame_i+1}/{TOTAL_FRAMES}]")
        sys.stdout.flush()
        
        # Sky background
        img = np.random.normal(loc=200, scale=10, size=(IMG_SIZE, IMG_SIZE)).astype(np.float32)
        
        # Inject the hardware hot pixels into the raw frame
        img[hot_ys, hot_xs] += hot_fluxes
        
        for i in range(TOTAL_STARS):
            x, y = all_xs[i], all_ys[i]
            flux = base_fluxes[i]
            
            if i in sn_idx:
                # Supernova logic: 0 before spawn, ramps up over 3 frames, then stays high
                if frame_i < event_frames[i]:
                    flux = 0 # Doesn't exist yet!
                elif frame_i == event_frames[i]:
                    flux *= 0.3 # 30% brightness
                elif frame_i == event_frames[i] + 1:
                    flux *= 0.7 # 70% brightness
                else:
                    flux *= 1.2 # Full brightness
            elif i in sflare_idx:
                if frame_i == event_frames[i]:
                    flux *= np.random.uniform(1.5, 2.0)
            elif i in oflare_idx:
                if frame_i == event_frames[i]:
                    flux *= np.random.uniform(5.0, 10.0)
            elif i in spulse_idx:
                flux *= 1.0 + 0.15 * np.sin(2 * np.pi * frame_i / 30.0)
            elif i in opulse_idx:
                flux *= 1.0 + 0.60 * np.sin(2 * np.pi * frame_i / 30.0)
                
            # Add star to image (if flux > 0)
            if flux > 0:
                psf_scaled = psf * flux
                psf_noisy = np.random.poisson(np.clip(psf_scaled, 0, None))
                sy, sx = psf_noisy.shape
                img[y-sy//2 : y+sy//2+1, x-sx//2 : x+sx//2+1] += psf_noisy
            
        filepath = os.path.join(SPOOL_DIR, f"sim_{frame_i:04d}.fits")
        fits.PrimaryHDU(img).writeto(filepath, overwrite=True)

def run_simulation():
    generate_hot_pixel_file()
    generate_simulated_data()
    
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)
    
    alerts_dir = os.path.join(OUT_DIR, "Alerts")
    
    # Monkeypatch Orchestrator to use our local mock hot pixel file
    def mock_getdata(path):
        return original_getdata(HOT_PIXEL_MOCK_PATH)
    def mock_exists(path):
        if path.endswith("hot_pixels.fts"): return True
        return original_exists(path)
    
    original_getdata = fits.getdata
    original_exists = os.path.exists
    
    # Actually just overwrite the path variable inside the orchestrator instance if we can
    # But it's hardcoded in the constructor. We can just patch the file reading locally.
    
    import builtins
    
    class MockOrchestrator(orchestrator.Orchestrator):
        def __init__(self, *args, **kwargs):
            # Hack the hot pixel path before calling super()
            # Wait, it's hardcoded inside the init.
            pass
            
    # Safer to just monkeypatch the os.path and fits dynamically during instantiation
    os.path.exists = mock_exists
    fits.getdata = mock_getdata
    
    # Instantiate
    daemon = orchestrator.Orchestrator(spool_directory=SPOOL_DIR)
    
    # Configure the logger to write to our OUT_DIR
    daemon.alert_logger.log_dir = alerts_dir
    if not os.path.exists(alerts_dir):
        os.makedirs(alerts_dir)
        
    import datetime
    def mock_log_alert(engine_name, x, y, full_image, crop_size=50, wcs=None):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        engine_prefix = "A" if "Engine A" in engine_name else "B"
        txt_path = os.path.join(daemon.alert_logger.log_dir, f"{timestamp}_{engine_prefix}_X{int(x)}_Y{int(y)}.txt")
        with open(txt_path, 'w') as f:
            f.write(f"Pixel: X={x}, Y={y}")
    daemon.alert_logger.log_alert = mock_log_alert
        
    # Restore
    os.path.exists = original_exists
    fits.getdata = original_getdata
        
    # Disable PNG generation for speed
    daemon.alert_logger.generate_png = False
    
    print("\nStarting Orchestrator Loop...")
    start_time = time.time()
    
    files = sorted(os.listdir(SPOOL_DIR))
    total_files = len([f for f in files if f.endswith('.fits')])
    processed = 0
    
    for f in files:
        if not f.endswith('.fits'): continue
        path = os.path.join(SPOOL_DIR, f)
        
        processed += 1
        sys.stdout.write(f"\rPipeline Processing... [{processed}/{total_files}]")
        sys.stdout.flush()
        
        daemon.process_new_image(path)
        
    end_time = time.time()
    
    # Grade Results
    print("\n" + "="*50)
    print("SIMULATION RESULTS")
    print("="*50)
    total_time = end_time - start_time
    print(f"Total Time: {total_time:.2f} s")
    print(f"Time per frame: {total_time / len(files):.4f} s")
    print(f"Frames processed: {len(files)}")
    
    # Load alerts
    detected_xs = []
    detected_ys = []
    if os.path.exists(alerts_dir):
        for fname in os.listdir(alerts_dir):
            if fname.endswith(".txt"):
                with open(os.path.join(alerts_dir, fname), 'r') as fh:
                    for line in fh:
                        if line.startswith("Pixel:"):
                            parts = line.strip().split(" ")
                            # "Pixel: X=543, Y=123"
                            dx = int(parts[1].split("=")[1].replace(",",""))
                            dy = int(parts[2].split("=")[1])
                            detected_xs.append(dx)
                            detected_ys.append(dy)
                            
    detected_xs = np.array(detected_xs)
    detected_ys = np.array(detected_ys)
    
    def check_detection(target_xs, target_ys):
        detected_count = 0
        for tx, ty in zip(target_xs, target_ys):
            if len(detected_xs) == 0: continue
            dists = np.sqrt((detected_xs - tx)**2 + (detected_ys - ty)**2)
            if np.min(dists) <= 5.0: # 5 pixel tolerance
                detected_count += 1
        return detected_count
        
    sflare_found = check_detection(all_xs[sflare_idx], all_ys[sflare_idx])
    oflare_found = check_detection(all_xs[oflare_idx], all_ys[oflare_idx])
    spulse_found = check_detection(all_xs[spulse_idx], all_ys[spulse_idx])
    opulse_found = check_detection(all_xs[opulse_idx], all_ys[opulse_idx])
    sn_found = check_detection(all_xs[sn_idx], all_ys[sn_idx])
    flat_found = check_detection(all_xs[flat_idx], all_ys[flat_idx])
    
    print("\n--- Detection Metrics ---")
    print(f"Slight Flares:     {sflare_found}/{NUM_SLIGHT_FLARE} Detected")
    print(f"Obvious Flares:    {oflare_found}/{NUM_OBVIOUS_FLARE} Detected")
    print(f"Slight Pulsators:  {spulse_found}/{NUM_SLIGHT_PULSE} Detected")
    print(f"Obvious Pulsators: {opulse_found}/{NUM_OBVIOUS_PULSE} Detected")
    print(f"Supernovae:        {sn_found}/{NUM_SUPERNOVA} Detected")
    
    print(f"\nFalse Positives (from {NUM_FLAT} flat stars): {flat_found}")
    print("="*50)
    
if __name__ == "__main__":
    run_simulation()
