import os
import sys
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# Ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from orchestrator import Orchestrator
from subtraction import optimal_image_subtraction

def create_mock_fits(filename, x, y, flux, shape=(100, 100), background=100.0, noise=5.0):
    """Generates a FITS image with exactly one star."""
    data = np.random.normal(background, noise, shape)
    
    # Create a 2D Gaussian point source (star)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * 1.5**2))
    data += flux * gaussian
    
    # Cast to float32 for pipeline
    data = data.astype(np.float32)
    
    hdu = fits.PrimaryHDU(data)
    hdu.writeto(filename, overwrite=True)

def main():
    spool_dir = "single_star_spool"
    if not os.path.exists(spool_dir):
        os.makedirs(spool_dir)
        
    print("Generating 10 frames with ONE star...")
    # Generate 10 frames. The star is at (50, 50).
    # Frame 1-6: Stable flux (500)
    # Frame 7: Flare! Flux jumps to 3000
    # Frame 8-10: Stable flux (500)
    for i in range(10):
        flux = 3000.0 if i == 6 else 500.0
        filename = os.path.join(spool_dir, f"frame_{i:02d}.fits")
        create_mock_fits(filename, 50, 50, flux)
        
    # Initialize Pipeline
    print("Initializing Pipeline...")
    orchestrator = Orchestrator(spool_dir)
    
    diff_image_to_plot = None
    
    try:
        for i in range(10):
            filepath = os.path.join(spool_dir, f"frame_{i:02d}.fits")
            print(f"Processing frame {i+1}/10...")
            orchestrator.process_new_image(filepath)
            
            # On the flare frame, capture the difference image
            if i == 6:
                with fits.open(filepath) as hdul:
                    target_image = hdul[0].data.astype(float)
                diff_image_to_plot, _ = optimal_image_subtraction(orchestrator.reference_image, target_image)
                
    except Exception as e:
        print(f"PIPELINE CRASHED: {e}")
        return

    print("Pipeline completed successfully! Generating plots...")
    
    # Plot 1: Light Curve
    # The star is the only star, so it's index 0 in the photometry engine
    artifacts_dir = "/Users/vihaa/.gemini/antigravity-ide/brain/0b3b5c28-8155-4813-bc9b-0f25a52964a9"
    lc_path = os.path.join(artifacts_dir, "single_star_lightcurve.png")
    
    if 0 in orchestrator.photometry_engine.light_curves:
        fluxes = orchestrator.photometry_engine.light_curves[0]
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(fluxes)+1), fluxes, marker='o', color='b')
        plt.title("Engine B: Photometric Light Curve of Single Star")
        plt.xlabel("Frame Number")
        plt.ylabel("Measured Flux (ADU)")
        plt.grid(True)
        plt.axvline(x=2, color='r', linestyle='--', label="Flare (Frame 7 overall, Frame 2 of Monitoring)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(lc_path, dpi=150)
        plt.close()
        print(f"Light curve saved to {lc_path}")
    else:
        print("ERROR: Photometry engine did not track the star!")

    # Plot 2: Difference Image
    diff_path = os.path.join(artifacts_dir, "single_star_difference.png")
    if diff_image_to_plot is not None:
        plt.figure(figsize=(6, 6))
        plt.imshow(diff_image_to_plot, origin='lower', cmap='viridis')
        plt.title("Engine A: Difference Image (Frame 7 Flare)")
        plt.colorbar(label='Residual Flux')
        plt.tight_layout()
        plt.savefig(diff_path, dpi=150)
        plt.close()
        print(f"Difference image saved to {diff_path}")
    else:
        print("ERROR: Difference image was not generated!")
        
    # Cleanup
    for file in os.listdir(spool_dir):
        os.remove(os.path.join(spool_dir, file))
    os.rmdir(spool_dir)
    print("Done!")

if __name__ == "__main__":
    main()
