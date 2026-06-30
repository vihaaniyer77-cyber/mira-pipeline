import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import numpy as np
from subtraction import optimal_image_subtraction, extract_sources_from_difference

def test_subtraction():
    ref = np.zeros((10, 10))
    ref[5, 5] = 100
    
    target = np.zeros((10, 10))
    target[5, 5] = 100
    target[2, 2] = 50 # transient
    
    diff = optimal_image_subtraction(target, ref)
    assert diff[2, 2] > 0
    # The center peak should be mostly subtracted
    
def test_extraction():
    diff = np.zeros((20, 20))
    diff[10, 10] = 500 # Strong peak
    
    # Needs to have some variance for background estimation
    diff += np.random.normal(0, 1, (20, 20))
    
    objects = extract_sources_from_difference(diff, background_sigma=5.0)
    assert len(objects) >= 1
    # Check if the peak is near 10,10
    found = False
    for obj in objects:
        if abs(obj['x'] - 10) < 2 and abs(obj['y'] - 10) < 2:
            found = True
    assert found
