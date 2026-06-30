import numpy as np
from starfinder import find_stars_autonomously

def test_starfinder():
    # Create a 50x50 blank image with a noise floor
    image = np.random.normal(10, 2, (50, 50))
    
    # Inject two bright "stars" (3x3 blob to pass DAOStarFinder sharpness filters)
    image[24:27, 24:27] += 30
    image[25, 25] += 70
    
    image[9:12, 39:42] += 30
    image[10, 40] += 70
    
    coords = find_stars_autonomously(image, fwhm_estimate=2.0, threshold_sigma=10.0)
    
    assert len(coords) == 2
    
    # Coordinates might not be exactly integer due to centroiding, but should be close
    x_coords = [c[0] for c in coords]
    y_coords = [c[1] for c in coords]
    
    # 25, 25 (Remember DAOStarFinder returns X, Y which corresponds to col, row)
    assert any(abs(x - 25) < 1.0 and abs(y - 25) < 1.0 for x, y in zip(x_coords, y_coords))
    
    # 40, 10 -> X=40, Y=10
    assert any(abs(x - 40) < 1.0 and abs(y - 10) < 1.0 for x, y in zip(x_coords, y_coords))
