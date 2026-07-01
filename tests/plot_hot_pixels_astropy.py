import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval

def main():
    dark_path = r"S:\Jean\Interns\Vihaan\hot_pixels.fts"
    
    if not os.path.exists(dark_path):
        print(f"Cannot find {dark_path}")
        return
        
    with fits.open(dark_path) as hdul:
        dark_data = hdul[0].data.astype(float)
        header = hdul[0].header
        
    # Extract values from header
    exptime = header.get('EXPTIME', 0.02)
    gain = header.get('EGAIN', 0.78)
    
    # The median represents the baseline bias/read noise level
    bias_level = np.median(dark_data)
    
    # ALGORITHM FROM ASTROPY GUIDE: Calculate dark current in e-/sec
    dark_current_e_sec = (dark_data - bias_level) * gain / exptime
    
    # The guide recommends a cutoff of around 1 e-/sec, but we can check the distribution.
    # Let's use 1.0 e-/sec as the threshold as recommended for typical cameras.
    hot_pixel_mask = dark_current_e_sec > 1.0
    
    num_hot = np.sum(hot_pixel_mask)
    print(f"Exposure Time: {exptime}s, Gain: {gain} e-/ADU")
    print(f"Bias Level: {bias_level:.2f} ADU")
    print(f"Found {num_hot} hot pixels out of {dark_data.size} ({num_hot/dark_data.size*100:.3f}%) using > 1.0 e-/sec dark current threshold.")
    
    # Plotting
    plt.figure(figsize=(10, 10))
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(dark_data)
    
    plt.imshow(dark_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    
    # Create an RGBA image for the overlay where only hot pixels are colored
    overlay = np.zeros((*dark_data.shape, 4))
    overlay[hot_pixel_mask] = [1.0, 0.0, 0.0, 1.0] # Red, fully opaque
    
    plt.imshow(overlay, origin='lower')
    
    plt.title(f"True Dark Current Map (> 1.0 e-/sec)\nIdentified {num_hot} Physical Hot Pixels")
    plt.axis('off')
    
    output_path = "hot_pixels_plot_astropy.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    main()
