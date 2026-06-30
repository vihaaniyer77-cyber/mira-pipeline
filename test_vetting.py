import numpy as np
from vetting import spatial_profile_vetting, TemporalVerifier

def test_vetting_spatial():
    # npix must be present as sep returns structured arrays with this field
    obj_good = {'a': 2.0, 'b': 1.8, 'npix': 10} # fwhm ~ 3.1, round
    obj_cosmic_ray = {'a': 0.5, 'b': 0.5, 'npix': 10} # fwhm ~ 1.4 (too small with corrected formula)
    obj_streak = {'a': 5.0, 'b': 1.0, 'npix': 10} # ellipticity = 0.8 (too elongated)
    
    assert spatial_profile_vetting(obj_good) == True
    assert spatial_profile_vetting(obj_cosmic_ray) == False
    assert spatial_profile_vetting(obj_streak) == False

def test_temporal_verifier():
    verifier = TemporalVerifier(required_consecutive=2)
    
    v1 = verifier.verify(['obj1'])
    assert len(v1) == 0
    
    v2 = verifier.verify(['obj1', 'obj2'])
    assert 'obj1' in v2
    assert 'obj2' not in v2
    
    v3 = verifier.verify(['obj2'])
    assert 'obj1' not in v3
    assert 'obj2' in v3
