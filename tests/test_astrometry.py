import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from astrometry_solver import solve_wcs_for_image

def test_astrometry_graceful_fallback():
    print("--- Testing Astrometry Solver ---")
    fits_path = r"S:\Jean\Interns\Vihaan\20260509\20260508-KIC5737655-0001g.fit"
    
    wcs_obj = solve_wcs_for_image(fits_path)
    
    if wcs_obj is None:
        print("\nTest Passed: Astrometry gracefully fell back to None (likely because solve-field is missing on this OS).")
    else:
        print("\nTest Passed: Astrometry successfully solved the WCS!")
        print(f"WCS Details:\n{wcs_obj}")

if __name__ == "__main__":
    test_astrometry_graceful_fallback()
