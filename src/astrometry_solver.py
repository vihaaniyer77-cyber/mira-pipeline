import os
import subprocess
from astropy.wcs import WCS
from astropy.io import fits

def solve_wcs_for_image(fits_filepath):
    """
    Attempts to solve the World Coordinate System (WCS) for a given FITS image using
    a local installation of astrometry.net (solve-field).
    
    
    Args:
        fits_filepath (str): Absolute path to the raw camera FITS file.
        
    Returns:
        wcs_object (astropy.wcs.WCS): The calculated WCS object, or None if it fails.
    """
    if not os.path.exists(fits_filepath):
        print(f"File not found: {fits_filepath}")
        return None
        
    base_path, _ = os.path.splitext(fits_filepath)
    wcs_output_path = base_path + ".wcs"
    
    try:
        # Run the local solve-field command -- launches an external terminal solve-field
        # --overwrite: Overwrite existing .wcs files
        # --no-plots: We don't need astrometry.net generating annotated images
        # --cpulimit 60: Fail fast if it can't solve it in 10 seconds
        result = subprocess.run(
            ["solve-field", "--overwrite", "--no-plots", "--cpulimit", "60", fits_filepath],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Check if the WCS file was successfully generated
        if os.path.exists(wcs_output_path):
            with fits.open(wcs_output_path) as wcs_hdul:
                # astropy.wcs reads the header directly
                wcs_object = WCS(wcs_hdul[0].header)
                print(f" Astrometry SUCCESS! WCS matrix locked for {os.path.basename(fits_filepath)}")
                
            # Clean up the output WCS file to save disk space
            os.remove(wcs_output_path)
            
            # Astrometry.net also sometimes generates a .new and .match file. Clean those up too.
            for ext in [".new", ".match", "-indx.xyls", ".axy"]:
                junk_file = base_path + ext
                if os.path.exists(junk_file):
                    os.remove(junk_file)
                    
            return wcs_object
            
        else:
            print(" Astrometry FAILED: solve-field could not match the stars. Falling back to X/Y pixels.")
            return None
            
    except FileNotFoundError:
        # solve-field is missing. Fall back to Astrometry.net Web API via astroquery
        api_key = os.environ.get("ASTROMETRY_API_KEY")
        if not api_key:
            print("Astrometry WARNING: 'solve-field' is not installed and ASTROMETRY_API_KEY is not set.")
            print("Falling back to raw X/Y pixel coordinates.")
            return None
            
        print("Astrometry WARNING: 'solve-field' not found. Falling back to Cloud API (Astrometry.net)...")
        from astroquery.astrometry_net import AstrometryNet
        ast = AstrometryNet()
        ast.api_key = api_key
        
        try:
            # Solve using the cloud API (this might take a few minutes)
            wcs_header = ast.solve_from_image(fits_filepath, force_image_upload=True, solve_timeout=300)
            if wcs_header:
                print(f" Cloud Astrometry SUCCESS! WCS matrix locked for {os.path.basename(fits_filepath)}")
                return WCS(wcs_header)
            else:
                print(" Cloud Astrometry FAILED: Server could not match the stars.")
                return None
        except Exception as api_e:
            print(f" Cloud Astrometry ERROR: {api_e}. Falling back to raw X/Y.")
            return None
            
    except Exception as e:
        print(f" Astrometry ERROR: {e}. Falling back to raw X/Y pixel coordinates.")
        return None
