import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ZScaleInterval
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture

def main():
    data_dir = r"S:\Jean\Interns\Vihaan\20260509"
    target = "KIC8508931"
    filter_band = "i"
    
    fits_files = sorted(glob.glob(os.path.join(data_dir, f"*{target}*{filter_band}.fit*")))
    if not fits_files:
        print("No files found.")
        return
        
    first_frame = fits_files[0]
    
    with fits.open(first_frame) as hdul:
        image = hdul[0].data.astype(float)
        
    # Physical Hot Pixel Mask
    hot_pixels_path = r"S:\Jean\Interns\Vihaan\hot_pixels.fts"
    if os.path.exists(hot_pixels_path):
        from scipy.ndimage import median_filter
        dark_data = fits.getdata(hot_pixels_path)
        bad_pixel_mask = dark_data > 700
        
        local_median = median_filter(image, size=3)
        image[bad_pixel_mask] = local_median[bad_pixel_mask]
        
    # Call the actual pipeline function to test the new KDTree crowding filter
    import sys
    sys.path.append('src')
    from starfinder import find_stars_autonomously
    
    positions = find_stars_autonomously(image, fwhm_estimate=3.0, threshold_sigma=5.0, max_stars=10000, min_separation=6.0)
    
    # Exactly what DAOStarFinder returns, now filtered by KDTree
    sources_len = len(positions)
    # EXACT mathematical aperture radius used in Engine B
    r = 3.0
    apertures = CircularAperture(positions, r=r)
    
    print(f"Found {sources_len} stars. Drawing true apertures...")
    
    # Zoom in on a dense 200x200 pixel crop to show exact mathematical scale
    crop_size = 200
    x_center, y_center = 2000, 2000 # Typical dense region
    
    plt.figure(figsize=(10, 10))
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(image)
    
    plt.imshow(image, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    
    # photutils has a built-in exact geometric plotter
    apertures.plot(color='red', lw=2.0, alpha=0.8)
    
    # Zoom into the crop
    plt.xlim(x_center - crop_size//2, x_center + crop_size//2)
    plt.ylim(y_center - crop_size//2, y_center + crop_size//2)
    
    plt.title(f"True Mathematical Apertures (Radius = {r} pixels)\nZoomed 200x200 crop")
    
    output_path = r"S:\Jean\Interns\Vihaan\true_aperture_scale.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved true scale comparison to {output_path}")

if __name__ == "__main__":
    main()
