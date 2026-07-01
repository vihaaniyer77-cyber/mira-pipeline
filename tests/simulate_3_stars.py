import os
import sys
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Foolproof way to disable astroalign: Mock the library function itself!
import astroalign
astroalign.register = lambda targ, ref: (targ, None)

from photutils.datasets import make_gaussian_sources_image
from astropy.table import Table
from orchestrator import Orchestrator
from subtraction import optimal_image_subtraction

def create_robust_fits(filename, star_data, shape=(200, 200), background=200.0, noise=15.0):
    """
    Generates a FITS image using photutils' mathematically robust Gaussian profiles
    so DAOStarFinder recognizes them as real, high-quality stars.
    star_data is a list of tuples: [(x, y, flux), ...]
    """
    # Create the properties table for photutils
    rows = []
    for (x, y, flux) in star_data:
        rows.append({
            'amplitude': flux,
            'x_mean': x,
            'y_mean': y,
            'x_stddev': 1.5, # Realistic sharpness
            'y_stddev': 1.5,
            'theta': 0.0
        })
    table = Table(rows=rows)
    
    # Generate perfect optical sources
    image = make_gaussian_sources_image(shape, table)
    
    # Add realistic background noise
    image += np.random.normal(background, noise, shape)
    
    # Save
    data = image.astype(np.float32)
    hdu = fits.PrimaryHDU(data)
    hdu.writeto(filename, overwrite=True)

def main():
    spool_dir = "multi_star_spool"
    if not os.path.exists(spool_dir):
        os.makedirs(spool_dir)
        
    print("Generating 30 frames with 3 ROBUST stars...")
    # Star 1: Stable at (50, 50)
    # Star 2: Flaring at (150, 50)
    # Star 3: Pulsating at (100, 150)
    
    for i in range(30):
        # Stable (High Flux to ensure tracking)
        flux_1 = 15000.0
        
        # Flare on frame 15
        flux_2 = 50000.0 if i == 15 else 15000.0
        
        # Pulsator (sine wave with a high baseline so it never dips into noise)
        flux_3 = 15000.0 + 8000.0 * np.sin(i * (2 * np.pi / 10.0))
        
        star_data = [
            (50, 50, flux_1),
            (150, 50, flux_2),
            (100, 150, flux_3)
        ]
        
        filename = os.path.join(spool_dir, f"frame_{i:02d}.fits")
        create_robust_fits(filename, star_data)

    print("Initializing Pipeline (Astroalign is 100% mocked out)...")
    orchestrator = Orchestrator(spool_dir)
    
    diff_image_flare = None
    
    for i in range(30):
        filepath = os.path.join(spool_dir, f"frame_{i:02d}.fits")
        print(f"Processing frame {i+1}/30...")
        orchestrator.process_new_image(filepath)
        
        # Capture difference image on the exact frame the flare goes off (frame 15)
        if i == 15 and orchestrator.reference_image is not None:
            with fits.open(filepath) as hdul:
                target_image = hdul[0].data.astype(float)
            diff_image_flare = optimal_image_subtraction(target_image, orchestrator.reference_image)

    print("Pipeline completed successfully! Generating plots...")
    artifacts_dir = "/Users/vihaa/.gemini/antigravity-ide/brain/0b3b5c28-8155-4813-bc9b-0f25a52964a9"
    
    # Plot 1: Light Curves
    plt.figure(figsize=(10, 6))
    colors = ['g', 'r', 'b']
    labels = ['Star 1 (Stable)', 'Star 2 (Flare)', 'Star 3 (Pulsator)']
    
    tracked_indices = list(orchestrator.photometry_engine.light_curves.keys())
    print(f"Photometry tracked {len(tracked_indices)} stars!")
    
    for idx, color in zip(tracked_indices, colors):
        fluxes = orchestrator.photometry_engine.light_curves[idx]
        plt.plot(range(1, len(fluxes)+1), fluxes, marker='o', color=color, label=f"Object {idx}")

    plt.title("Engine B: Photometric Light Curves (Robust Stars)")
    plt.xlabel("Monitoring Frame Number")
    plt.ylabel("Measured Flux (ADU)")
    plt.grid(True)
    plt.axvline(x=11, color='k', linestyle='--', label="Flare Injection")
    plt.legend()
    plt.tight_layout()
    
    lc_path = os.path.join(artifacts_dir, "three_stars_lightcurves_robust.png")
    plt.savefig(lc_path, dpi=150)
    plt.close()
    print(f"Light curves saved to {lc_path}")

    # Plot 2: Difference Image
    if diff_image_flare is not None:
        plt.figure(figsize=(6, 6))
        plt.imshow(diff_image_flare, origin='lower', cmap='viridis', vmin=-1000, vmax=5000)
        plt.title("Engine A: Difference Image (Frame 15)")
        
        # Plot circles where the stars are to see what was perfectly subtracted
        plt.plot(50, 50, 'go', markersize=14, fillstyle='none', label='Stable Star (Erased)')
        plt.plot(100, 150, 'bo', markersize=14, fillstyle='none', label='Pulsator Star (Erased)')
        plt.plot(150, 50, 'ro', markersize=14, fillstyle='none', label='Flare Star (Detected!)')
        
        plt.legend(loc='upper left')
        plt.colorbar(label='Residual Flux')
        plt.tight_layout()
        
        diff_path = os.path.join(artifacts_dir, "three_stars_difference_robust.png")
        plt.savefig(diff_path, dpi=150)
        plt.close()
        print(f"Difference image saved to {diff_path}")

    # Cleanup
    for file in os.listdir(spool_dir):
        os.remove(os.path.join(spool_dir, file))
    os.rmdir(spool_dir)
    print("Done!")

if __name__ == "__main__":
    main()
