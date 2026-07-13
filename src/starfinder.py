import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.spatial import cKDTree

def find_stars_autonomously(image, fwhm_estimate=3.0, threshold_sigma=5.0, max_stars=2000, saturation_level=55000.0, min_separation=6.0):
 
    # Estimate the background and background noise
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    
    # https://photutils.readthedocs.io/en/stable/user_guide/index.html?__cf_chl_f_tk=nCEvkYnwI47vL8uPS2VDGbWkpUYCSbQ13nkrgW8C240-1783358576-1.0.1.1-Ioz4b5Z.o94sDQmV5ijRN.kaBAfrbOX_MRopAMHCuKU
    daofind = DAOStarFinder(fwhm=fwhm_estimate, threshold=threshold_sigma * std, peakmax=saturation_level, sharplo=0.2, sharphi=0.8)
    
    # 3. Execute the search
    sources = daofind(image - median)
    
    if sources is None or len(sources) == 0:
        return []
        
    # Sort by brightest stars first
    sources.sort('flux')
    sources.reverse()

    raw_coords = np.array([(row['xcentroid'], row['ycentroid']) for row in sources])
    
    #  KDTree Distance Filter (Crowding Contamination)
    # Identify pairs of stars that are closer than min_separation
    tree = cKDTree(raw_coords)
    pairs = tree.query_pairs(min_separation)
    
    # Conservative rejection policy: discard BOTH stars in a crowded pair.
    # With aperture_radius=3.0px and min_separation=6.0px, any pair returned here
    # has genuinely overlapping apertures, meaning both flux measurements are
    # contaminated. We prefer fewer clean light curves over more blended ones.
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
