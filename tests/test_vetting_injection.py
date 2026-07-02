import os
import sys
import numpy as np
from astropy.io import fits
import sep

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from vetting import saturation_vetting, spatial_profile_vetting
from subtraction import optimal_image_subtraction, extract_sources_from_difference

def test_injection():
    # 1. Create a fake background
    bg_level = 200.0
    ref_image = np.random.normal(bg_level, 5.0, (200, 200)).astype(np.float32)
    target_image = np.copy(ref_image)
    
    # 2. Inject a saturated star
    sat_x, sat_y = 50, 50
    # Create a bleeding column
    target_image[sat_y-5:sat_y+5, sat_x] = 65000.0
    target_image[sat_y, sat_x-2:sat_x+3] = 65000.0
    
    # 3. Inject a valid transient (e.g. a supernova)
    sn_x, sn_y = 150, 150
    # 2D Gaussian
    y, x = np.mgrid[0:200, 0:200]
    sn_profile = 5000.0 * np.exp(-((x - sn_x)**2 + (y - sn_y)**2) / (2 * 2.0**2))
    target_image += sn_profile.astype(np.float32)
    
    # 4. Inject a cosmic ray (sharp, 1 pixel)
    cr_x, cr_y = 100, 100
    target_image[cr_y, cr_x] += 10000.0
    
    # 5. Run Engine A (Subtraction)
    # We pass a simple identity kernel to avoid solving for it in the fake noise
    psf_kernel = np.zeros((5, 5))
    psf_kernel[2, 2] = 1.0
    diff_image = optimal_image_subtraction(target_image, ref_image, psf_kernel=psf_kernel)
    
    # 6. Extract Sources
    objects = extract_sources_from_difference(diff_image, background_sigma=5.0)
    print(f"Engine A found {len(objects)} raw objects.")
    
    # 7. Run Vetting
    survivors = []
    for obj in objects:
        x_obj, y_obj = obj['x'], obj['y']
        
        # Spatial Profile Vetting
        if not spatial_profile_vetting(obj):
            print(f"Rejected at ({x_obj:.1f}, {y_obj:.1f}) -> Failed Spatial Vetting (CR/Streak)")
            continue
            
        # Saturation Vetting
        if not saturation_vetting(x_obj, y_obj, target_image, saturation_level=60000.0):
            print(f"Rejected at ({x_obj:.1f}, {y_obj:.1f}) -> Failed Saturation Vetting (Bleeding Star)")
            continue
            
        survivors.append(obj)
        
    print(f"Vetting complete. {len(survivors)} objects survived.")
    for s in survivors:
        print(f"Survivor at ({s['x']:.1f}, {s['y']:.1f}) with shape a={s['a']:.2f}, b={s['b']:.2f}, npix={s['npix']}")
        
    assert len(survivors) == 1, "Should only have 1 survivor (the valid supernova)"
    assert abs(survivors[0]['x'] - sn_x) < 1.0, "Survivor is not the supernova"
    
    print("Injection test passed! The vetting system correctly isolated the true transient.")

if __name__ == "__main__":
    test_injection()
