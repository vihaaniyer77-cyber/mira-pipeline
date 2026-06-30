import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import time
import shutil
import numpy as np
from astropy.io import fits
from orchestrator import Orchestrator
from starfinder import find_stars_autonomously
import pytest

# Use a temporary directory for stress testing
STRESS_DIR = "/Users/vihaa/.gemini/antigravity-ide/scratch/transient_pipeline/stress_spool"

@pytest.fixture(scope="module")
def setup_stress_dir():
    if os.path.exists(STRESS_DIR):
        shutil.rmtree(STRESS_DIR)
    os.makedirs(STRESS_DIR)
    yield STRESS_DIR
    shutil.rmtree(STRESS_DIR)

def create_fits(filename, data):
    hdu = fits.PrimaryHDU(data)
    filepath = os.path.join(STRESS_DIR, filename)
    hdu.writeto(filepath, overwrite=True)
    return filepath

def test_crowded_field_cpu_cap():
    """
    Test 1: The 'Crowded Field' CPU Test
    Inject 5,000 noise peaks. Ensure max_stars limits the extraction to prevent CPU freezing.
    """
    # 500x500 image with tons of "stars"
    np.random.seed(42)
    image = np.random.normal(10, 2, (500, 500))
    
    # Inject 5,000 stars randomly
    for _ in range(5000):
        x = np.random.randint(5, 495)
        y = np.random.randint(5, 495)
        image[y, x] += 1000
        image[y-1:y+2, x-1:x+2] += 200

    start_time = time.time()
    stars = find_stars_autonomously(image, max_stars=2000)
    end_time = time.time()
    
    # Assert CPU time is reasonable (under 5 seconds)
    assert (end_time - start_time) < 5.0
    # Assert cap was enforced
    assert len(stars) <= 2000

def test_long_shift_memory_endurance(setup_stress_dir):
    """
    Test 2: The 'Long Shift' Memory Endurance Test
    Process 50 frames to ensure light_curves and TemporalVerifier history truncate properly.
    """
    orchestrator = Orchestrator()
    
    # Create 3 stars
    base_data = np.random.normal(10, 2, (100, 100))
    base_data[20:23, 20:23] += 500
    base_data[50:53, 50:53] += 500
    base_data[80:83, 80:83] += 500
    
    # Feed 50 frames
    for i in range(50):
        data = base_data.copy() + np.random.normal(0, 2, (100, 100))
        filepath = create_fits(f"mem_{i:03d}.fits", data)
        orchestrator.process_new_image(filepath)
    
    # Verify memory limits
    # light_curves history should not exceed 2 * window_size (20)
    for star_id, history in orchestrator.photometry_engine.light_curves.items():
        assert len(history) <= 20
        
    # Temporal bouncer history should only contain active objects
    assert len(orchestrator.temporal_verifier.history) <= 0 # No transients were injected!

def test_massive_outbreak_io(setup_stress_dir):
    """
    Test 3: The 'Massive Outbreak' I/O Test
    Inject 20 flares simultaneously to ensure AlertLogger handles disk I/O without crashing.
    """
    orchestrator = Orchestrator()
    
    # Phase 1: Burn-In (5 frames)
    base_data = np.random.normal(10, 2, (200, 200))
    
    # Map 20 stars
    for s in range(20):
        base_data[10+(s*8):13+(s*8), 10+(s*8):13+(s*8)] += 500
        
    for i in range(5):
        filepath = create_fits(f"io_{i:02d}.fits", base_data)
        orchestrator.process_new_image(filepath)
        
    # Phase 2: Frame 6 (Normal)
    filepath = create_fits(f"io_05.fits", base_data)
    orchestrator.process_new_image(filepath)
    
    # Phase 3: Frame 7 (MASSIVE OUTBREAK - All 20 stars flare!)
    outbreak_data = base_data.copy()
    for s in range(20):
        outbreak_data[11+(s*8), 11+(s*8)] += 10000 # Massive flare
        
    filepath = create_fits(f"io_06.fits", outbreak_data)
    
    start_time = time.time()
    orchestrator.process_new_image(filepath)
    end_time = time.time()
    
    # Should process and log 20 flares in under 5 seconds
    assert (end_time - start_time) < 5.0
    
def test_thick_clouds_graceful_failure(setup_stress_dir):
    """
    Test 4: The 'Thick Clouds' Graceful Failure Test
    Pass pure noise (no stars). Ensure pipeline resets or ignores gracefully without crashing.
    """
    orchestrator = Orchestrator()
    
    # Pure noise (clouds covering everything)
    noise = np.random.normal(10, 10, (100, 100))
    
    # This should fail alignment/burn-in gracefully
    try:
        for i in range(6):
            filepath = create_fits(f"clouds_{i:02d}.fits", noise)
            orchestrator.process_new_image(filepath)
    except Exception as e:
        pytest.fail(f"Pipeline crashed on pure noise: {e}")
