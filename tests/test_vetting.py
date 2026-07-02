import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from vetting import saturation_vetting, spatial_profile_vetting, TemporalVerifier

def test_saturation_vetting():
    print("--- Testing Saturation Vetting ---")
    # Create an empty 10x10 image
    img = np.zeros((10, 10))
    # Add a saturated pixel
    img[5, 5] = 61000.0
    
    # Check at exactly the saturated pixel
    assert saturation_vetting(5, 5, img) == False, "Failed to reject directly on saturated pixel"
    # Check 1 pixel away
    assert saturation_vetting(6, 5, img) == False, "Failed to reject near saturated pixel"
    # Check far away
    assert saturation_vetting(1, 1, img) == True, "Rejected safe pixel"
    
    print("Saturation vetting tests passed!")

def test_spatial_profile_vetting():
    print("--- Testing Spatial Profile Vetting ---")
    # Valid round star
    valid_star = {'a': 2.0, 'b': 1.9, 'npix': 10}
    assert spatial_profile_vetting(valid_star) == True, "Rejected valid star"
    
    # Highly elliptical (satellite streak)
    streak = {'a': 5.0, 'b': 1.0, 'npix': 20}
    assert spatial_profile_vetting(streak) == False, "Failed to reject streak"
    
    # Too small (hot pixel / cosmic ray)
    cosmic_ray = {'a': 0.2, 'b': 0.2, 'npix': 1}
    assert spatial_profile_vetting(cosmic_ray) == False, "Failed to reject tiny cosmic ray"
    
    print("Spatial profile vetting tests passed!")

def test_temporal_verifier():
    print("--- Testing Temporal Verifier ---")
    verifier = TemporalVerifier(required_consecutive=3, tolerance=2.0)
    
    # Frame 1: Star at (10.0, 10.0)
    survivors = verifier.verify([(10.0, 10.0)])
    assert len(survivors) == 0, "Alerted too early"
    
    # Frame 2: Star jiggles to (10.5, 10.2) -> dist < 2.0, so it matches
    survivors = verifier.verify([(10.5, 10.2)])
    assert len(survivors) == 0, "Alerted too early"
    
    # Frame 3: Star jiggles to (9.8, 10.1) -> 3rd frame! Should trigger
    survivors = verifier.verify([(9.8, 10.1)])
    assert len(survivors) == 1 and survivors[0] == (9.8, 10.1), "Failed to track jittering star"
    
    # Frame 4: Star completely disappears (e.g. cloud or fake transient)
    survivors = verifier.verify([])
    assert len(survivors) == 0, "Alerted when star disappeared"
    
    # Frame 5: Star returns at (9.8, 10.1), count should have been reset!
    survivors = verifier.verify([(9.8, 10.1)])
    assert len(survivors) == 0, "Count did not reset when star disappeared!"
    
    # Ensure two distinct objects can be tracked
    verifier = TemporalVerifier(required_consecutive=3, tolerance=2.0)
    verifier.verify([(5.0, 5.0), (20.0, 20.0)])
    verifier.verify([(5.0, 5.0), (20.0, 20.0)])
    survivors = verifier.verify([(5.0, 5.0), (20.0, 20.0)])
    assert len(survivors) == 2, "Failed to track multiple stars simultaneously"
    
    print("Temporal verifier tests passed!")

if __name__ == "__main__":
    test_saturation_vetting()
    test_spatial_profile_vetting()
    test_temporal_verifier()
