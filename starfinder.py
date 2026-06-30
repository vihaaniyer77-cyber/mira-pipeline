import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

def find_stars_autonomously(image, fwhm_estimate=3.0, threshold_sigma=5.0, max_stars=2000, saturation_level=55000.0):
    """
    Scans an image to dynamically locate the (X, Y) pixel coordinates of all 
    stars in the field of view. This acts as the autonomous mapping phase 
    for Engine B when no historical catalog is available.
    
    Args:
        image: 2D numpy array (typically the pristine Dynamic Reference image)
        fwhm_estimate: The estimated Full Width at Half Maximum of stars in pixels.
                       Depends on the telescope's typical seeing.
        threshold_sigma: How many standard deviations above the background noise 
                         a source must be to be considered a star.
        max_stars: The maximum number of stars to track. CPU OVERLOAD PATCH.
        saturation_level: ADU threshold to ignore flat-topped/bleeding stars.
                         
    Returns:
        List of (x, y) tuples representing the exact centroids of the found stars.
    """
    # 1. Estimate the background and background noise
    # We use sigma clipping to ignore the actual bright stars while finding the noise floor
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    
    # 2. Initialize the DAOStarFinder algorithm
    # threshold = background + (threshold_sigma * standard_deviation)
    daofind = DAOStarFinder(fwhm=fwhm_estimate, threshold=threshold_sigma * std, peakmax=saturation_level)
    
    # 3. Execute the search
    # Subtracting median background to ensure we are only detecting true peaks
    sources = daofind(image - median)
    
    # If no stars are found (e.g. extremely thick clouds or dome is closed)
    if sources is None:
        return []
        
    # Sort by brightest stars first and enforce the max_stars limit
    sources.sort('peak')
    sources.reverse()
    sources = sources[:max_stars]
    
    # 4. Extract the exact X and Y centroids
    coordinates = []
    for row in sources:
        x_centroid = row['xcentroid']
        y_centroid = row['ycentroid']
        coordinates.append((x_centroid, y_centroid))
        
    return coordinates
