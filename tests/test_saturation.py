import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import numpy as np
from starfinder import find_stars_autonomously
from vetting import saturation_vetting

def test_saturation_starfinder():
    # Create an image with a normal star and a saturated star
    image = np.random.normal(10, 2, (100, 100))
    
    # Normal star (Peak = 3000)
    image[20:23, 20:23] += 1000
    image[21, 21] += 2000
    
    # Saturated star (Peak = 60000)
    image[80:83, 80:83] += 10000
    image[81, 81] += 50000
    
    # Run starfinder with saturation_level=55000
    stars = find_stars_autonomously(image, saturation_level=55000.0)
    
    # It should only find 1 star! (The saturated one should be ignored)
    assert len(stars) == 1
    
    # The found star should be near (21, 21)
    x, y = stars[0]
    assert np.isclose(x, 21, atol=2)
    assert np.isclose(y, 21, atol=2)

def test_saturation_vetting():
    # Create a raw image with a saturated pixel
    raw_image = np.zeros((50, 50))
    
    # A saturated pixel at (25, 25)
    raw_image[25, 25] = 60000
    
    # Test checking exactly on the saturated pixel
    assert saturation_vetting(25, 25, raw_image, saturation_level=55000) == False
    
    # Test checking adjacent to the saturated pixel (within search_radius=2)
    assert saturation_vetting(26, 26, raw_image, saturation_level=55000) == False
    
    # Test checking far away (safe)
    assert saturation_vetting(10, 10, raw_image, saturation_level=55000) == True
