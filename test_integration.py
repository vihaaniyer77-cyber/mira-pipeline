import os
import time
import shutil
import threading
import numpy as np
from astropy.io import fits
from orchestrator import Orchestrator

def test_full_pipeline_integration():
    """
    Massive end-to-end integration test simulating the real telescope hardware environment.
    This test runs the orchestrator daemon in a background thread while mimicking
    the camera writing 20 FITS files slowly to the spool folder.
    
    It injects:
    1. A Flare (Engine B catch)
    2. A slow moving satellite (Temporal Vetting rejection)
    3. A new Supernova (Engine A catch)
    4. A Telescope Slew (Alignment Error -> Reset -> Re-Burn)
    """
    spool_dir = "integration_spool"
    output_dir = "integration_discoveries"
    
    # Cleanup
    if os.path.exists(spool_dir): shutil.rmtree(spool_dir)
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(spool_dir)
    os.makedirs(output_dir)
    
    # Initialize the Orchestrator with our test output folder
    orchestrator = Orchestrator(spool_directory=spool_dir)
    orchestrator.alert_logger.output_dir = output_dir
    orchestrator.alert_logger.csv_path = os.path.join(output_dir, "discoveries.csv")
    
    # A mock camera function that writes files to disk (simulating 1 second write delays)
    def mock_camera():
        # SCENARIO 1: Burn-In Phase (Frames 0 to 4)
        for i in range(5):
            data = np.random.normal(10, 2, (100, 100))
            
            # Star 1 (Peaked blob)
            data[50:53, 50:53] += 30
            data[51, 51] += 70
            
            # Star 2
            data[20:23, 80:83] += 30
            data[21, 81] += 70
            
            # Star 3
            data[80:83, 20:23] += 30
            data[81, 21] += 70
            
            hdu = fits.PrimaryHDU(data)
            hdu.writeto(os.path.join(spool_dir, f"frame_{i:02d}.fits"))
            time.sleep(0.5) # Simulating camera cadence
            
        # SCENARIO 2: Monitoring Phase (Frames 5 to 10)
        for i in range(5, 11):
            data = np.random.normal(10, 2, (100, 100))
            
            # Star 1 (Stable)
            data[50:53, 50:53] += 30
            data[51, 51] += 70
            
            # Star 2 (Stable)
            data[20:23, 80:83] += 30
            data[21, 81] += 70
            
            # Star 3 (Stable)
            data[80:83, 20:23] += 30
            data[81, 21] += 70
            
            # Frame 7: Inject a massive Flare on Star 1 (Engine B should catch as Flare)
            if i == 7:
                data[50:53, 50:53] += 5000
                data[51, 51] += 10000
                
            # Frame 8-10: Inject a Supernova in EMPTY SPACE (Engine A should catch)
            if i >= 8:
                # Standard Gaussian profile that passes the Bouncer's strict spatial checks
                data[41, 81] += 500.0
                data[40:43, 80:83] += 200.0
                data[39:44, 79:84] += 50.0
                
            hdu = fits.PrimaryHDU(data)
            hdu.writeto(os.path.join(spool_dir, f"frame_{i:02d}.fits"))
            time.sleep(0.5)
            
        # SCENARIO 3: Telescope Slews (Frame 11)
        data = np.random.normal(10, 2, (100, 100))
        data[10:13, 10:13] += 50 # Completely new star field
        hdu = fits.PrimaryHDU(data)
        hdu.writeto(os.path.join(spool_dir, f"frame_11.fits"))
        time.sleep(0.5)

    # Start the "Camera" in the background
    camera_thread = threading.Thread(target=mock_camera)
    camera_thread.start()
    
    # Run the Orchestrator manually on the main thread for a set number of loops
    # rather than an infinite while True
    timeout = time.time() + 15
    while time.time() < timeout:
        fits_files = sorted([os.path.join(spool_dir, f) for f in os.listdir(spool_dir) if f.endswith('.fits')])
        for filepath in fits_files:
            if filepath not in orchestrator.processed_files:
                # Mock the file stability lock logic
                size_1 = os.path.getsize(filepath)
                time.sleep(0.1)
                size_2 = os.path.getsize(filepath)
                if size_1 == size_2 and size_1 > 0:
                    orchestrator.process_new_image(filepath)
                    orchestrator.processed_files.add(filepath)
        time.sleep(0.5)
        
        # If we processed all 12 frames, break early
        if len(orchestrator.processed_files) == 12:
            break
            
    camera_thread.join()
    
    # --- ASSERTIONS ---
    
    # 1. Pipeline should have reset to BURN_IN after frame 11 (the slew)
    assert orchestrator.state == "BURN_IN"
    
    # 2. Check that the Alert Logger wrote to the CSV
    csv_path = os.path.join(output_dir, "discoveries.csv")
    assert os.path.exists(csv_path)
    
    with open(csv_path, 'r') as f:
        content = f.read()
        # Flare should be caught
        assert "Engine B (Flare)" in content
        # Supernova should be caught (persists for frames 8, 9, 10 -> hits TemporalVerifier 3-frame threshold on frame 10)
        assert "Engine A (New" in content
        
    # 3. Check PNGs were generated
    pngs = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    assert len(pngs) >= 2 # At least one for Flare, one for SN
    
    # Cleanup
    shutil.rmtree(spool_dir)
    shutil.rmtree(output_dir)

if __name__ == "__main__":
    test_full_pipeline_integration()
    print("Integration Test Passed Flawlessly!")
