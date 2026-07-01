import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, mad_std
from astropy.visualization import ZScaleInterval
import matplotlib.colors as mcolors

def main():
    dark_path = r"S:\Jean\Interns\Vihaan\hot_pixels.fts"
    
    if not os.path.exists(dark_path):
        print(f"Cannot find {dark_path}")
        return
        
    with fits.open(dark_path) as hdul:
        dark_data = hdul[0].data.astype(float)
        
    # Standard astropy CCD reduction guide technique for hot pixels
    # Calculate robust statistics
    median_dark = np.median(dark_data)
    mad_std_dark = mad_std(dark_data)
    
    print(f"Dark Median: {median_dark:.2f}, MAD Std: {mad_std_dark:.2f}")
    
    # Identify hot pixels as those > median + 5*MAD (or similar threshold)
    threshold = median_dark + 5.0 * mad_std_dark
    hot_pixel_mask = dark_data > threshold
    
    num_hot = np.sum(hot_pixel_mask)
    print(f"Threshold: {threshold:.2f}")
    print(f"Found {num_hot} hot pixels out of {dark_data.size} ({num_hot/dark_data.size*100:.3f}%)")
    
    # Plotting
    plt.figure(figsize=(10, 10))
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(dark_data)
    
    # Create a custom colormap to overlay hot pixels in red
    plt.imshow(dark_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    
    # Create an RGBA image for the overlay where only hot pixels are colored
    overlay = np.zeros((*dark_data.shape, 4))
    overlay[hot_pixel_mask] = [1.0, 0.0, 0.0, 1.0] # Red, fully opaque
    
    plt.imshow(overlay, origin='lower')
    
    plt.title(f"Hot Pixel Map overlay on Dark Frame\nIdentified {num_hot} Hot Pixels (>5σ)")
    plt.axis('off')
    
    output_path = "hot_pixels_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    main()
