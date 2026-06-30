import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
import numpy as np
from astrometry_solver import solve_wcs_for_image
from alert_logger import AlertLogger

def test_astrometry_fallback_missing_file():
    """
    Test that the solver gracefully returns None when given a bad filepath,
    rather than crashing the orchestrator loop.
    """
    wcs = solve_wcs_for_image("non_existent_file.fits")
    assert wcs is None

def test_alert_logger_fallback():
    """
    Test that the AlertLogger can handle wcs=None without throwing an exception,
    falling back to logging RA/Dec as 'Unknown'.
    """
    logger = AlertLogger(output_dir="test_astrometry_output")
    
    # Mock image
    fake_image = np.random.normal(10, 2, (100, 100))
    
    # Should not crash
    logger.log_alert("Engine A (Test)", 50, 50, fake_image, wcs=None)
    
    # Verify the CSV was created and contains 'Unknown' for RA/Dec
    csv_path = os.path.join("test_astrometry_output", "discoveries.csv")
    assert os.path.exists(csv_path)
    
    with open(csv_path, 'r') as f:
        content = f.read()
        assert "Unknown" in content
        
    # Cleanup
    for file in os.listdir("test_astrometry_output"):
        os.remove(os.path.join("test_astrometry_output", file))
    os.rmdir("test_astrometry_output")
