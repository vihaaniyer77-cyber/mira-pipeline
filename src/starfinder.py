import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.spatial import cKDTree

def find_stars_autonomously(image, fwhm_estimate=3.0, threshold_sigma=5.0, max_stars=2000, saturation_level=55000.0, min_separation=6.0):
    """
    Scans an image to dynamically locate the (X, Y) pixel coordinates of all 
    stars in the field of view, excluding those that are too crowded.
    Args:
        image: 2D numpy array (typically the pristine Dynamic Reference image)
        fwhm_estimate: The estimated Full Width at Half Maximum of stars in pixels.
        threshold_sigma: How many standard deviations above the background noise 
                         a source must be to be considered a star.
        max_stars: The maximum number of stars to track. CPU OVERLOAD PATCH.
        saturation_level: ADU threshold to ignore flat-topped/bleeding stars.
        min_separation: The minimum distance in pixels allowed between two stars.
                         If closer than this, BOTH stars are rejected to prevent 
                         crowding contamination.
                         
    Returns:
        List of (x, y) tuples representing the exact centroids of the found stars.
    """
    # 1. Estimate the background and background noise
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    
    # 2. Initialize the DAOStarFinder algorithm
    daofind = DAOStarFinder(fwhm=fwhm_estimate, threshold=threshold_sigma * std, peakmax=saturation_level, sharplo=0.2, sharphi=0.8)
    
    # 3. Execute the search
    sources = daofind(image - median)
    
    if sources is None or len(sources) == 0:
        return []
        
    # Sort by brightest stars first
    sources.sort('peak')
    sources.reverse()
    
    # 4. Extract raw coordinates
    raw_coords = np.array([(row['xcentroid'], row['ycentroid']) for row in sources])
    
    # 5. KDTree Distance Filter (Crowding Contamination)
    # Identify pairs of stars that are closer than min_separation
    tree = cKDTree(raw_coords)
    pairs = tree.query_pairs(min_separation)
    
    # Build a set of all indices that are part of ANY close pair (reject both)
    crowded_indices = set()
    for i, j in pairs:
        crowded_indices.add(i)
        crowded_indices.add(j)
        
    # 6. Filter and enforce max_stars limit
    isolated_coords = []
    for i in range(len(raw_coords)):
        if i not in crowded_indices:
            isolated_coords.append(tuple(raw_coords[i]))
            if len(isolated_coords) >= max_stars:
                break
                
    return isolated_coords 
