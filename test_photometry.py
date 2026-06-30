import numpy as np
from photometry import PhotometryEngine

def test_photometry_zscore():
    # Use min_std=0 so the test exercises real anomaly detection
    # (production uses min_std>0 to suppress noise false positives)
    engine = PhotometryEngine(window_size=5, z_threshold=3.0, min_std=0.0)
    
    # Feed 5 normal frames
    for _ in range(5):
        z, stds, z_alerts, var_alerts = engine.update_light_curves([100.0, 50.0])
        assert len(z_alerts) == 0
        
    # Feed an anomaly
    z, stds, z_alerts, var_alerts = engine.update_light_curves([200.0, 50.0]) # source 0 flares
    assert 0 in z_alerts
    assert 1 not in z_alerts
    assert z[0] > 3.0

def test_photometry_variance():
    # min_std = 15.0, var_threshold_multiplier = 3.0 => threshold is 45.0
    engine = PhotometryEngine(window_size=5, min_std=15.0, var_threshold_multiplier=3.0)
    
    # Simulate a high variance pulsator
    for i in range(5):
        flux = 100.0 + 100.0 * np.sin(i * 2.0 * np.pi / 4.0)
        z, stds, z_alerts, var_alerts = engine.update_light_curves([flux])
        
    assert 0 in var_alerts
    assert stds[0] > 45.0
