import os
import subprocess
from astropy.wcs import WCS
from astropy.io import fits

def solve_wcs_for_image(fits_filepath):
    """
    Attempts to solve the World Coordinate System (WCS) for a given FITS image using
    a local installation of astrometry.net (solve-field).
    
    If the software is not installed, or the field cannot be solved, this function 
    fails gracefully and returns None, triggering the pipeline's fallback mechanism.
    
    Args:
        fits_filepath (str): Absolute path to the raw camera FITS file.
        
    Returns:
        wcs_object (astropy.wcs.WCS): The calculated WCS object, or None if it fails.
    """
    if not os.path.exists(fits_filepath):
        print(f"File not found: {fits_filepath}")
        return None
        
    wcs_output_path = fits_filepath.replace(".fits", ".wcs")
    
    try:
        # Run the local solve-field command
        # --overwrite: Overwrite existing .wcs files
        # --no-plots: We don't need astrometry.net generating annotated images
        # --cpulimit 10: Fail fast if it can't solve it in 10 seconds
        result = subprocess.run(
            ["solve-field", "--overwrite", "--no-plots", "--cpulimit", "10", fits_filepath],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Check if the WCS file was successfully generated
        if os.path.exists(wcs_output_path):
            with fits.open(wcs_output_path) as wcs_hdul:
                # astropy.wcs reads the header directly
                wcs_object = WCS(wcs_hdul[0].header)
                print(f"✅ Astrometry SUCCESS! WCS matrix locked for {os.path.basename(fits_filepath)}")
                
            # Clean up the output WCS file to save disk space
            os.remove(wcs_output_path)
            
            # Astrometry.net also sometimes generates a .new and .match file. Clean those up too.
            for ext in [".new", ".match", "-indx.xyls", ".axy"]:
                junk_file = fits_filepath.replace(".fits", ext)
                if os.path.exists(junk_file):
                    os.remove(junk_file)
                    
            return wcs_object
            
        else:
            print("⚠️ Astrometry FAILED: solve-field could not match the stars. Falling back to X/Y pixels.")
            return None
            
    except FileNotFoundError:
        # This triggers if "solve-field" is not installed on the system (e.g. during local tests)
        print("Astrometry WARNING: 'solve-field' is not installed or not in PATH.")
        print("Falling back to raw X/Y pixel coordinates. Please install astrometry.net for RA/Dec.")
        return None
    except Exception as e:
        print(f"⚠️ Astrometry ERROR: {e}. Falling back to raw X/Y pixel coordinates.")
        return None
