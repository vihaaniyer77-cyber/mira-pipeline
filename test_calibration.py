import numpy as np
import pytest
from calibration import calibrate_image, align_image, generate_master_reference, AlignmentError

def test_calibration():
    # Mock data
    raw = np.ones((10, 10)) * 100
    bias = np.ones((10, 10)) * 10
    dark = np.ones((10, 10)) * 5
    flat = np.ones((10, 10)) * 0.5
    
    calibrated = calibrate_image(raw, bias, dark, flat, exposure_time=2.0, dark_exposure=1.0)
    # raw - bias = 90
    # scaled_dark = 5 * (2/1) = 10
    # 90 - 10 = 80
    # 80 / 0.5 = 160
    assert np.allclose(calibrated, 160.0)

def test_alignment_error():
    # Test that without stars, it raises AlignmentError indicating a slew
    target = np.random.rand(10, 10)
    ref = np.random.rand(10, 10)
    with pytest.raises(AlignmentError):
        align_image(target, ref)

def test_dynamic_reference():
    # 3 frames of background 100
    frames = [np.ones((10, 10)) * 100 for _ in range(3)]
    
    # Introduce a massive artifact into frame 1
    frames[1][5, 5] = 9999
    
    ref = generate_master_reference(frames)
    
    # Median should reject the artifact completely
    assert ref[5, 5] == 100
    assert np.array_equal(ref, np.ones((10, 10)) * 100)
