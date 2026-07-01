import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from photutils.datasets import make_random_gaussians_table, make_gaussian_sources_image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from starfinder import find_stars_autonomously

def main():
    # 1. Generate a highly realistic mock star field
    # 500 stars, varying brightness, with FWHM ~ 3.0
    print("Generating realistic optical star field...")
    shape = (500, 500)
    table = make_random_gaussians_table(
        n_sources=500,
        param_ranges={
            'amplitude': [500, 20000],
            'x_mean': [10, 490],
            'y_mean': [10, 490],
            'x_stddev': [1.2, 1.8],
            'y_stddev': [1.2, 1.8],
            'theta': [0, np.pi]
        }
    )
    
    # Create the image and add realistic Gaussian background noise
    image = make_gaussian_sources_image(shape, table)
    background_noise = np.random.normal(200.0, 15.0, shape)
    image += background_noise
    
    # 2. Run our pipeline's exact autonomous mapping algorithm
    print("Running MIRA Pipeline DAOStarFinder...")
    try:
        # Our function expects a 2D numpy array
        tracked_stars_xy = find_stars_autonomously(image)
        print(f"Success! DAOStarFinder mapped {len(tracked_stars_xy)} stable sources.")
    except Exception as e:
        print(f"Error during star finding: {e}")
        return

    # 3. Generate the Quality Assurance (QA) Visual Plot
    print("Generating Quality Assurance visual plot...")
    
    plt.figure(figsize=(10, 10))
    
    # ZScale interval mimics how astronomers view FITS files (auto-stretches contrast)
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(image)
    
    plt.imshow(image, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    plt.title(f"DAOStarFinder QA: {len(tracked_stars_xy)} Sources Tracked")
    
    # Plot red circles over every tracked star
    for (x, y) in tracked_stars_xy:
        plt.plot(x, y, 'ro', markersize=14, fillstyle='none', markeredgewidth=1.5, alpha=0.7)
        
    plt.tight_layout()
    
    artifacts_dir = "/Users/vihaa/.gemini/antigravity-ide/brain/0b3b5c28-8155-4813-bc9b-0f25a52964a9"
    plot_path = os.path.join(artifacts_dir, "starfinder_qa_plot.png")
    
    plt.savefig(plot_path, dpi=200)
    plt.close()
    
    print(f"Visual QA plot saved successfully to {plot_path}")

if __name__ == "__main__":
    main()
