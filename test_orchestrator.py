import os
import shutil
import numpy as np
from astropy.io import fits
from orchestrator import Orchestrator

def test_orchestrator():
    spool_dir = "test_spool"
    
    if os.path.exists(spool_dir):
        shutil.rmtree(spool_dir)
    os.makedirs(spool_dir)
    
    orchestrator = Orchestrator(spool_directory=spool_dir)
    
    # 1. Create 7 mock FITS files to simulate an observation session
    # We will make them all identical so astroalign succeeds
    for i in range(7):
        data = np.random.normal(10, 2, (100, 100))
        # Add a "star" so astroalign can align them
        data[50:53, 50:53] += 50
        data[20:23, 80:83] += 50
        data[80:83, 20:23] += 50
        
        hdu = fits.PrimaryHDU(data)
        filepath = os.path.join(spool_dir, f"image_{i}.fits")
        hdu.writeto(filepath)
        
        # Manually process them instead of running the infinite watchdog
        orchestrator.process_new_image(filepath)
        
    # Check assertions
    # The first 5 should trigger a state change to MONITORING
    assert orchestrator.state == "MONITORING"
    assert orchestrator.reference_image is not None
    assert len(orchestrator.background_stars_xy) > 0
    
    # Files 6 and 7 were processed through the engines
    # Verify that the TemporalVerifier saw something (at least the static stars)
    # Actually, Engine A shouldn't find anything because the images are identical
    # But the state should be clean.
    
    # 2. Simulate a telescope slew by passing a completely different image
    # (Stars in different places)
    slew_data = np.random.normal(10, 2, (100, 100))
    slew_data[10:13, 10:13] += 50 
    slew_hdu = fits.PrimaryHDU(slew_data)
    slew_path = os.path.join(spool_dir, "image_slew.fits")
    slew_hdu.writeto(slew_path)
    
    orchestrator.process_new_image(slew_path)
    
    # The slew should have triggered an AlignmentError and reset the pipeline!
    assert orchestrator.state == "BURN_IN"
    assert orchestrator.reference_image is None
    assert len(orchestrator.burn_in_cache) == 1 # The slew image is now frame 1 of new burn-in
    
    shutil.rmtree(spool_dir)
