#UPDATED
#UPDATE 2
# MIRA Pipeline Standalone
import os
import sys
import csv
import time
import glob
import logging
import threading
import datetime
import subprocess
import argparse
import collections
from collections import deque

import numpy as np
import sep
import astroalign as aa
import scipy.fft
from scipy.signal import fftconvolve
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ZScaleInterval
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry

# --- DYNAMIC DRIVE DETECTION ---
def get_extreme_ssd_path():
    import ctypes
    try:
        for drive in range(ord('A'), ord('Z')+1):
            drive_letter = chr(drive) + ':\\\\'
            if os.path.exists(drive_letter):
                volume_name_buf = ctypes.create_unicode_buffer(1024)
                ctypes.windll.kernel32.GetVolumeInformationW(
                    ctypes.c_wchar_p(drive_letter),
                    volume_name_buf,
                    ctypes.sizeof(volume_name_buf),
                    None, None, None, None, 0
                )
                if 'Extreme SSD' in volume_name_buf.value:
                    return drive_letter
    except Exception as e:
        print(f'Failed to query drives: {e}')
    return None


# ============================================================
# src/alert_logger.py
# ============================================================

class AlertLogger:
  
    def __init__(self, output_dir=None):
        if output_dir is None:
            self.output_dir = os.path.join(os.path.expanduser("~"), "Desktop", "pipeline_discoveries")
        else:
            self.output_dir = output_dir
            
        self.csv_path = os.path.join(self.output_dir, "discoveries.csv")
       
        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
           
        self._init_csv()

    def _init_csv(self):
        # Initialize the CSV with headers if it doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Engine", "Significance", "X_Pixel", "Y_Pixel", "RA", "Dec", "Image_File"])

    def rotate_csv(self):
        """Called on a telescope slew to archive the old discoveries and start fresh."""
        if os.path.exists(self.csv_path):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = os.path.join(self.output_dir, f"discoveries_{timestamp}.csv")
            try:
                os.rename(self.csv_path, archive_path)
                logging.info(f"Rotated previous discoveries to {archive_path}")
            except Exception as e:
                logging.error(f"Failed to rotate CSV: {e}")
        self._init_csv()

    def log_alert(self, engine_name, x, y, full_image, crop_size=50, wcs=None, significance="Unknown"):
        
        # Ensure integers for indexing
        x, y = int(round(x)), int(round(y))
       
        # 1. Generate a unique timestamp filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        engine_short = engine_name.split()[1].replace("(", "").replace(")", "") if " " in engine_name else engine_name
        txt_filename = f"{timestamp}_{engine_short}_X{x}_Y{y}.txt"
        txt_filepath = os.path.join(self.output_dir, txt_filename)
       
        # 2. Handle Astrometry (WCS Transformation)
        ra_str = "Unknown"
        dec_str = "Unknown"
        if wcs is not None:
            try:
                sky_coord = wcs.pixel_to_world(x, y)
                ra_str = f"{sky_coord.ra.deg:.5f}"
                dec_str = f"{sky_coord.dec.deg:.5f}"
               
                # Write an explicit text file so the astronomer can copy-paste RA/Dec
                with open(txt_filepath, 'w') as f:
                    f.write(f"--- MIRA PIPELINE ALERT ---\n")
                    f.write(f"Type: {engine_name}\n")
                    f.write(f"Significance: {significance}\n")
                    f.write(f"RA (deg): {ra_str}\n")
                    f.write(f"Dec (deg): {dec_str}\n")
                    f.write(f"Pixel: X:{x}, Y:{y}\n")
            except Exception as e:
                print(f"Warning: Failed to convert pixels to RA/Dec: {e}")
               
        # 5. Append to the localized CSV database
        try:
            with open(self.csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, engine_name, significance, x, y, ra_str, dec_str, txt_filename])
        except Exception as e:
            print(f"Warning: Failed to write to CSV database: {e}")
           
        # 6. Console Notification
        print(f"DISCOVERY LOGGED [{engine_name} | {significance}] at RA:{ra_str}, Dec:{dec_str} (X:{x}, Y:{y}). Saved to {txt_filename}")

# ============================================================
# src/astrometry_solver.py
# ============================================================

def solve_wcs_for_image(fits_filepath):
    
    if not os.path.exists(fits_filepath):
        print(f"File not found: {fits_filepath}")
        return None
        
    base_path, _ = os.path.splitext(fits_filepath)
    wcs_output_path = base_path + ".wcs"
    #IMPORTANT, MAKE SURE TO DOWNLOAD THE FILES BEFOREHAND!!!!!!
    try:
        # Convert Windows path to WSL Linux path (e.g. C:\... -> /mnt/c/...)
        wsl_path_result = subprocess.run(
            ["wsl", "wslpath", "-a", "-u", fits_filepath],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if wsl_path_result.returncode == 0 and wsl_path_result.stdout.strip():
            linux_filepath = wsl_path_result.stdout.strip()
        else:
            linux_filepath = fits_filepath  # Fallback just in case

        # Run the local solve-field command -- launches an external terminal solve-field
        # --overwrite: Overwrite existing .wcs files
        # --no-plots: We don't need astrometry.net generating annotated images
        # --cpulimit 60: Fail fast if it can't solve in 60 seconds
        # --scale-units arcsecperpix / --scale-low / --scale-high: Provide telescope scale hints to speed up solving
        result = subprocess.run(
            ["wsl", "solve-field", "--overwrite", "--no-plots", "--cpulimit", "60", 
             "--scale-units", "arcsecperpix", "--scale-low", "0.59", "--scale-high", "0.62", 
             linux_filepath],
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
                
            # Clean up all output files generated by solve-field to save disk space
            for ext in [".wcs", ".new", ".match", "-indx.xyls", ".axy", ".rdls", ".corr", ".solved"]:
                junk_file = base_path + ext
                if os.path.exists(junk_file):
                    os.remove(junk_file)
                    
            return wcs_object
            
        else:
            print(f" Astrometry FAILED: solve-field could not match the stars. stderr: {result.stderr[:200]}")
            print(" Falling back to X/Y pixels.")
            return None
            
    except FileNotFoundError:
        # solve-field is missing. Fall back to Astrometry.net Web API via astroquery
        api_key = os.environ.get("ASTROMETRY_API_KEY")
        if not api_key:
            print("Astrometry WARNING: 'solve-field' is not installed and ASTROMETRY_API_KEY is not set.")
            print("Falling back to raw X/Y pixel coordinates.")
            return None
            
        print("Astrometry WARNING: 'solve-field' not found. Falling back to Cloud API (Astrometry.net)...")
        print("WARNING: Cloud solve will block the pipeline for up to 120 seconds.")
        from astroquery.astrometry_net import AstrometryNet
        ast = AstrometryNet()
        ast.api_key = api_key
        
        try:
            # Solve using the cloud API.
            # NOTE: This is a blocking call. If local solve-field is unavailable on a
            # production run, consider running this in a separate thread to avoid
            # freezing the daemon loop for the duration of the cloud solve.
            wcs_header = ast.solve_from_image(fits_filepath, force_image_upload=True, solve_timeout=120)
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

# ============================================================
# src/vetting.py
# ============================================================

def saturation_vetting(x, y, raw_image, saturation_level=55000.0, search_radius=2):
    
    x, y = int(round(x)), int(round(y))
    
    # Define bounding box, ensuring we don't index outside the image
    y_min = max(0, y - search_radius)
    y_max = min(raw_image.shape[0], y + search_radius + 1)
    x_min = max(0, x - search_radius)
    x_max = min(raw_image.shape[1], x + search_radius + 1)
    
    # Check if ANY pixel in this local patch exceeds the saturation limit
    patch = raw_image[y_min:y_max, x_min:x_max]
    if np.any(patch >= saturation_level):
        return False # Reject: Artifact of saturation
        
    return True # Safe

def spatial_profile_vetting(extracted_object, min_fwhm=2.0, max_fwhm=8.0, max_ellipticity=0.4, min_pixels=4):
    
    a = extracted_object['a']
    b = extracted_object['b']
    
    try:
        npix = extracted_object['npix']
    except (ValueError, KeyError):
        npix = min_pixels  # safe fallback
    
    # Reject things that are too small or mathematically invalid
    if a <= 0 or npix < min_pixels:
        return False
    
    # Mathematical FWHM approximation (assuming a Gaussian Point Spread Function)
    # The standard astronomical conversion from a Gaussian sigma to FWHM is 2.3548
    # sep returns the semi-major/minor axes 'a' and 'b' as the sigma of the profile
    fwhm = 2.0 * np.sqrt(2.0 * np.log(2)) * (a + b) / 2.0
    
    # Ellipticity (1 - b/a). 
    # High ellipticity indicates a satellite streak or optical tracking smear.
    # True stellar transients must be round (ellipticity close to 0).
    ellipticity = 1.0 - (b / a)
    
    is_valid_fwhm = min_fwhm <= fwhm <= max_fwhm
    is_valid_shape = 0.0 <= ellipticity <= max_ellipticity
    
    return is_valid_fwhm and is_valid_shape

class TemporalVerifier:
    
    def __init__(self, required_consecutive=3, tolerance=2.0, baseline_frames=10):
        self.required = required_consecutive
        self.tolerance = tolerance
        self.baseline_frames = baseline_frames
        self.frame_count = 0
        self.history = {} # obj_id -> {'count': int, 'last_pos': (x, y), 'alerted': bool, 'missed': int}
        self.next_id = 0
        
    def verify(self, current_detections_xy):
        self.frame_count += 1
       
        valid_targets = []
        matched_ids = set()
        
        # Link current detections to history using nearest neighbor within tolerance
        for cx, cy in current_detections_xy:
            best_match_id = None
            best_dist = float('inf')
            
            for obj_id, data in self.history.items():
                if obj_id in matched_ids:
                    continue # One-to-one mapping
                    
                hx, hy = data['last_pos']
                dist = np.sqrt((cx - hx)**2 + (cy - hy)**2)
                
                if dist < self.tolerance and dist < best_dist:
                    best_match_id = obj_id
                    best_dist = dist
                    
            if best_match_id is not None:
                # Update existing track and reset miss counter
                self.history[best_match_id]['count'] += 1
                self.history[best_match_id]['last_pos'] = (cx, cy)
                self.history[best_match_id]['missed'] = 0
                matched_ids.add(best_match_id)
                
                # Only fire an alert the first time a target crosses the threshold.
                # Without this, a persistent transient (nova, slow flare) would generate
                # a new alert PNG on every single frame indefinitely.
                # The flag resets naturally when the object disappears (its history is deleted).
                if self.history[best_match_id]['count'] >= self.required:
                    if not self.history[best_match_id]['alerted']:
                        valid_targets.append((cx, cy))
                        self.history[best_match_id]['alerted'] = True
            else:
                # Create new track
                self.history[self.next_id] = {'count': 1, 'last_pos': (cx, cy), 'alerted': False, 'missed': 0}
                matched_ids.add(self.next_id)
                self.next_id += 1
                
        # For objects not seen this frame, increment their miss counter.
        # Allow 1 missed frame before deleting the track — a single bad frame
        # (cloud, cosmic ray, brief seeing spike) should not reset a genuine detection.
        # MEMORY LEAK PATCH: Delete the key entirely once miss tolerance is exceeded.
        for obj_id in list(self.history.keys()):
            if obj_id not in matched_ids:
                self.history[obj_id]['missed'] += 1
                if self.history[obj_id]['missed'] > 1:
                    del self.history[obj_id]
                
        # Enforce the baseline wait: no alerts are allowed until the baseline is established
        if self.frame_count <= self.baseline_frames:
            return []
            
        return valid_targets

# ============================================================
# src/starfinder.py
# ============================================================

def find_stars_autonomously(image, fwhm_estimate=3.0, threshold_sigma=5.0, max_stars=2000, saturation_level=55000.0, min_separation=16.0):
    
    # Estimate the background and background noise
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    
    # https://photutils.readthedocs.io/en/stable/user_guide/index.html?__cf_chl_f_tk=nCEvkYnwI47vL8uPS2VDGbWkpUYCSbQ13nkrgW8C240-1783358576-1.0.1.1-Ioz4b5Z.o94sDQmV5ijRN.kaBAfrbOX_MRopAMHCuKU
    daofind = DAOStarFinder(fwhm=fwhm_estimate, threshold=threshold_sigma * std, peakmax=saturation_level, sharplo=0.2, sharphi=0.8)
    
    # 3. Execute the search
    sources = daofind(image - median)
    
    if sources is None or len(sources) == 0:
        return []
        
    # Sort by brightest stars first
    sources.sort('flux')
    sources.reverse()

    raw_coords = np.array([(row['xcentroid'], row['ycentroid']) for row in sources])
    
    #  KDTree Distance Filter (Crowding Contamination)
    # Identify pairs of stars that are closer than min_separation
    tree = cKDTree(raw_coords)
    pairs = tree.query_pairs(min_separation)
    

    crowded_indices = set()
    for i, j in pairs:
        crowded_indices.add(i)
        crowded_indices.add(j)
        
    # 6. Filter and enforce max_stars limit
    isolated_coords = []
    for i in range(len(raw_coords)):
        if i not in crowded_indices:
            isolated_coords.append(tuple(raw_coords[i]))
            if len(isolated_coords) >= max_stars:
                break
                
    return isolated_coords

# ============================================================
# src/subtraction.py
# ============================================================

def fit_optimal_kernel(target, reference, kernel_size=5):
    """
    Solves the Alard-Lupton optimal kernel matching equation.
    Because atmospheric blurring ('seeing') changes constantly, we cannot simply
    subtract two images. This function calculates a spatial convolution kernel (K) 
    that mathematically matches the point spread function (PSF) of the reference 
    image to the target image.
    
    It minimizes the least-squares difference: (target - reference ⊗ K)^2
    
    Args:
        target: 2D numpy array (the current camera frame)
        reference: 2D numpy array (the dynamic burn-in reference)
        kernel_size: Integer size of the matching kernel matrix (default 5x5)
        
    Returns:
        K: The 2D convolution matrix that models the atmospheric difference.
    """
    half_k = kernel_size // 2
    
    y_min, x_min = half_k, half_k
    y_max, x_max = reference.shape[0] - half_k, reference.shape[1] - half_k
    
    stride = 10
    I_flat = target[y_min:y_max:stride, x_min:x_max:stride].flatten()
    
    num_pixels = I_flat.shape[0]
    M = np.zeros((num_pixels, kernel_size**2))
    
    col = 0
    for i in range(-half_k, half_k + 1):
        for j in range(-half_k, half_k + 1):
            # Use -i, -j to correctly match the mathematical definition of convolution
            # (which fftconvolve uses) rather than cross-correlation.
            patch = reference[y_min-i : y_max-i : stride, x_min-j : x_max-j : stride]
            M[:, col] = patch.flatten()
            col += 1
            
    # Add a column of ones to M to solve for the differential background
    M_bg = np.hstack([M, np.ones((M.shape[0], 1))])
    
    # Ridge penalty injected into the diagonal to prevent matrix singularity
    # (Ensures stability even if parts of the image are perfectly black)
    ridge = 1e-4 * np.eye(kernel_size**2)
    
    # Expand ridge penalty to include background (background gets 0 penalty)
    ridge_bg = np.zeros((kernel_size**2 + 1, kernel_size**2 + 1))
    ridge_bg[:kernel_size**2, :kernel_size**2] = ridge
    
    # INITIAL FIT
    sol = np.linalg.solve(M_bg.T @ M_bg + ridge_bg, M_bg.T @ I_flat)
    
    # --- SIGMA-CLIPPING: Exclude transient pixels from the kernel solve ---
    # We must clip based on the RESIDUALS of the initial fit, NOT the raw image!
    # Clipping the raw image throws away all the stars, which the solver needs to match PSFs.
    residuals = I_flat - (M_bg @ sol)
    mad = np.median(np.abs(residuals - np.median(residuals)))
    robust_std = 1.4826 * mad
    
    if robust_std > 0:
        good_mask = np.abs(residuals) < 5.0 * robust_std
        if good_mask.sum() >= kernel_size ** 2 + 1:
            I_flat = I_flat[good_mask]
            M_bg = M_bg[good_mask, :]
            # RE-FIT with outliers excluded
            sol = np.linalg.solve(M_bg.T @ M_bg + ridge_bg, M_bg.T @ I_flat)

    
    # Extract kernel and background offset
    k_flat = sol[:-1]
    bg_diff = sol[-1]
    
    K = k_flat.reshape((kernel_size, kernel_size))
    
    # NOTE: The forced normalization K /= K.sum() has been removed.
    # The kernel must be allowed to sum to <1 or >1 to properly scale
    # the reference image if the target image has different atmospheric transmission (clouds).
    
    return K, bg_diff

def optimal_image_subtraction(target_image, reference_image, psf_kernel=None, bg_diff=0.0):
    """
    Engine A: The Discovery Engine.
    
    This engine hunts for completely uncataloged objects (like a new supernova)
    that appear in empty space. It dynamically blurs the pristine reference image 
    to match the atmospheric distortion of the current frame, then subtracts them.
    
    Math: Difference = Target - (Reference ⊗ K) - Bkg
    
    Returns:
        difference_image: A 2D array where static stars have been mathematically 
                          erased, leaving only pure noise and new transients.
    """
    if psf_kernel is None:
        # Calculate the dynamic atmospheric blur and background offset in BOTH directions
        # This handles the case where the reference image is blurrier than the target image,
        # preventing massive ringing artifacts from unstable deconvolution.
        K1, bg1 = fit_optimal_kernel(target_image, reference_image, kernel_size=5)
        K2, bg2 = fit_optimal_kernel(reference_image, target_image, kernel_size=5)
        
        # Less negative mass means a more physically stable blurring kernel (closer to 0 is better).
        if np.sum(K1[K1 < 0]) > np.sum(K2[K2 < 0]):
            # Reference is sharper. Blur reference to match target.
            with scipy.fft.set_workers(-1):
                convolved_ref = fftconvolve(reference_image, K1, mode='same')
            difference_image = target_image - convolved_ref - bg1
        else:
            # Target is sharper. Blur target to match reference.
            with scipy.fft.set_workers(-1):
                convolved_target = fftconvolve(target_image, K2, mode='same')
            difference_image = convolved_target - reference_image - bg2
            
    else:
        # Artificially blur the reference image using Fast Fourier Transform convolution (Multi-Core)
        with scipy.fft.set_workers(-1):
            convolved_ref = fftconvolve(reference_image, psf_kernel, mode='same')
        
        # Subtract to isolate transients, applying the background offset
        difference_image = target_image - convolved_ref - bg_diff
    
    return difference_image

def extract_sources_from_difference(difference_image, background_sigma=20.0, edge_margin=250):
    """
    Scans the subtracted difference image to find statistically significant clusters
    of glowing pixels that survived the subtraction process.
    
    Args:
        difference_image: 2D numpy array (the output of Engine A)
        background_sigma: The SNR threshold required to trigger an extraction.
                          (e.g. 5.0 means the object must be 5x brighter than the noise floor)
        edge_margin: Number of pixels from the edge of the image to ignore.
                     (Ignores artifacts from zero-padding during image alignment)
                          
    Returns:
        objects: A structured numpy array of detections (includes 'x', 'y', 'a', 'b', 'flux').
                 These are the raw transient candidates sent to the Vetting Bouncer.
        globalrms: The global background RMS noise floor (used for significance calculation).
    """
    # Cast to float64 and ensure native byte order — SEP's C backend requires
    # native-endian arrays, and FITS files are big-endian by default.
    diff_data = np.ascontiguousarray(difference_image, dtype=np.float64)
    
    # Dynamically estimate the background RMS (noise floor) of the subtracted image
    bkg = sep.Background(diff_data)
    
    # Calculate the extraction threshold.
    # 20-sigma is intentionally aggressive. The primary target event class is supernovae,
    # which are extremely luminous and produce screaming detections on a difference image.
    # A high threshold prevents subtraction residuals near bright stars from flooding the
    # vetting pipeline with false positives, especially since there is no downstream ML classifier.
    thresh = background_sigma * bkg.globalrms
    
    # Increase deblending limit to handle crowded stellar fields.
    # The default of 1024 is often exceeded on dense star fields.
    sep.set_sub_object_limit(4096)
    
    try:
        objects = sep.extract(diff_data - bkg.back(), thresh)
    except Exception as e:
        # Graceful fallback: if deblending still overflows, raise the threshold
        # and try again at 30-sigma to only grab the brightest survivors
        try:
            objects = sep.extract(diff_data - bkg.back(), 30.0 * bkg.globalrms)
        except Exception:
            # If it still fails, return an empty array
            import numpy.lib.recfunctions as rfn
            objects = np.array([], dtype=[('x','f4'),('y','f4'),('a','f4'),('b','f4'),('flux','f4'),('npix','i4'),('peak','f4')])
    
    if len(objects) > 0:
        h, w = difference_image.shape
        # Filter out objects too close to the edge to avoid alignment padding artifacts
        good_mask = (
            (objects['x'] > edge_margin) & (objects['x'] < w - edge_margin) &
            (objects['y'] > edge_margin) & (objects['y'] < h - edge_margin)
        )
        objects = objects[good_mask]
        
    return objects, bkg.globalrms

# ============================================================
# src/calibration.py
# ============================================================

class AlignmentError(Exception):
    """Raised when astroalign fails, usually indicating the telescope slewed."""
    pass

def calibrate_image(raw_image, master_bias, master_flat, bad_pixel_mask=None):
    """Applies basic CCD calibration (bias subtraction, flat division, bad pixel healing)."""
    # Subtract bias (readout noise)
    calibrated = raw_image.astype(float) - master_bias
    
    # Divide by flat (vignetting & dust spots)
    # Zero-guard to prevent divide-by-zero on edge pixels
    flat_safe = np.where(master_flat == 0, 1.0, master_flat)
    calibrated /= flat_safe
    
    # Heal hot/dead pixels if a mask is provided
    if bad_pixel_mask is not None:
        if bad_pixel_mask.shape != calibrated.shape:
            print("Warning: bad_pixel_mask shape does not match image shape. Skipping mask.")
        else:
            local_median = median_filter(calibrated, size=3)
            calibrated[bad_pixel_mask] = local_median[bad_pixel_mask]
            
    return calibrated

def apply_calibration(raw_image, hot_pixels=None, flat_field=None, bias_frame=None):
    """
    Applies flat-fielding, bias subtraction, and hot pixel correction.
    """
    calibrated = np.copy(raw_image).astype(np.float32)
    
    if bias_frame is not None:
        calibrated = calibrated - bias_frame
        
    if hot_pixels is not None:
        calibrated[hot_pixels > 0] = np.median(calibrated)
    
    if flat_field is not None:
        calibrated = calibrated / flat_field
        
    return calibrated

def align_image(target_image, reference_image):

    try:
        aligned, _ = aa.register(target_image, reference_image)
        return aligned
    except aa.MaxIterError:
        raise AlignmentError("Astroalign failed. The telescope likely slewed to a new target.")
    except Exception as e:
        raise AlignmentError(f"Unexpected alignment failure: {str(e)}")


def generate_master_reference(burn_in_frames):
    """Creates a clean, deep reference image by aligning and median-stacking the burn-in frames."""
    if not burn_in_frames:
        raise ValueError("Must provide at least one frame for burn-in.")
        
    anchor = burn_in_frames[0]
    aligned_frames = [anchor]
    
    # Align all subsequent frames to the first frame
    for frame in burn_in_frames[1:]:
        try:
            aligned = align_image(frame, anchor)
            aligned_frames.append(aligned)
        except AlignmentError:
            print("Warning: A burn-in frame failed alignment. Skipping it.")
            # It's fine if one doesn't work, we just need a decent stack
            
    if len(aligned_frames) < 3:
        # We  want at least 3 frames for a proper median stack to reject outliers
        print("Building reference frame with fewer than 3 images. . . ")
        
    stack = np.array(aligned_frames)
    master_reference = np.median(stack, axis=0)
    
    return master_reference

# ============================================================
# src/photometry.py
# ============================================================

class PhotometryEngine:
    
    def __init__(self, z_threshold=5.0, min_std=25.0, var_threshold_multiplier=5.0):
        self.z_threshold = z_threshold
        self.var_threshold_multiplier = var_threshold_multiplier
        self.light_curves = {} # source_id (index) -> list of fluxes
        self.reference_fluxes = {} # source_id -> raw flux on first frame
        self.baselines = {} # source_id -> (baseline_mean, baseline_std)
        self.already_alerted_var = set() # Prevent infinite spam for pulsators
        
    def perform_aperture_photometry(self, image, positions, aperture_radius=8.0):
        """
        Uses photutils to extract rapid aperture photometry for thousands of stars simultaneously.
        
        Args:
            image: 2D numpy array of the current camera frame.
            positions: list of (x, y) tuples provided by DAOStarFinder.
            
        Returns:
            1D numpy array of flux values, indexed to match the input positions list.
        """
        if not positions:
            return []
            
        # Dynamically subtract sky background.
        # Cast to float64 and ensure native byte order — SEP's C backend requires
        # native-endian arrays, and FITS files are big-endian by default.
        img_data = np.ascontiguousarray(image, dtype=np.float64)
        bkg = sep.Background(img_data)
        bkg_subtracted = img_data - bkg.back()
            
        apertures = CircularAperture(positions, r=aperture_radius)
        phot_table = aperture_photometry(bkg_subtracted, apertures)
        return phot_table['aperture_sum'].value

    def update_light_curves(self, fluxes):
        """
        Maintains an in-memory time series of flux values for every tracked star.
        Evaluates the rolling Z-score and rolling variance for the current frame.
        
        Args:
            fluxes: array-like of flux values corresponding to the fixed source IDs.
            
        Returns:
            z_scores: array of current Z-scores for all stars.
            stds: array of current rolling standard deviations for all stars.
            z_alerts: list of source_ids that triggered a Flare alert (Z-score > threshold).
            var_alerts: list of source_ids that triggered a Variable alert (Variance > threshold).
        """
        z_scores = []
        stds = []
        z_alerts = []
        var_alerts = []
        
        # Store reference fluxes on the very first frame
        if not self.reference_fluxes:
            for i, flux in enumerate(fluxes):
                self.reference_fluxes[i] = flux
        
        # Calculate the global ensemble zero-point correction
        ratios = []
        for i, flux in enumerate(fluxes):
            ref_flux = self.reference_fluxes.get(i, 0.0)
            # Only use bright, stable stars for the ensemble calculation
            if ref_flux > 100.0:
                ratios.append(flux / ref_flux)
        
        # Determine the global flux correction factor (median of all valid ratios)
        # If no history exists yet (first frame), factor is 1.0
        if len(ratios) >= 5:
            correction_factor = np.median(ratios)
        else:
            correction_factor = 1.0
            
        # Guard against zero or NaN
        if not np.isfinite(correction_factor) or correction_factor <= 0:
            correction_factor = 1.0
            
        # Apply the correction to all incoming fluxes (Ensemble Differential Photometry)
        corrected_fluxes = [f / correction_factor for f in fluxes]
        
        # --- Pass 1: update histories, compute z-scores and relative variances ---
        # MIN_PULSATOR_FRAMES: minimum history length before the variance trigger is eligible.
        # Requires substantially more frames than the 10-frame baseline window so that the
        # variance estimate is statistically meaningful (uncertainty ∝ 1/√N).
        # At 30 frames the variance estimate uncertainty is ~26%; at 15 frames it is ~38%.
        # This also prevents the check from firing during the early part of a short run
        # where genuine pulsation cannot be distinguished from noise.
        MIN_PULSATOR_FRAMES = 30

        relative_vars = {}  # source_id -> Var(history) / b_mean^2  (dimensionless)

        for i, flux in enumerate(corrected_fluxes):
            if i not in self.light_curves:
                self.light_curves[i] = []

            history = self.light_curves[i]
            history.append(flux)

            if len(history) == 10:
                # Lock in the baseline statistics after the burn-in window
                self.baselines[i] = (np.mean(history), np.std(history))

            if len(history) >= 10:
                b_mean, b_std = self.baselines[i]

                # Protect against zero std; apply 3% systematic noise floor
                b_std = max(b_std, 10.0, 0.03 * abs(b_mean))
                stds.append(b_std)

                z = (flux - b_mean) / b_std

                # Trigger 1: Flare / Instantaneous Outlier Check
                # Using absolute value catches both positive spikes (flares) and
                # sudden deep negative spikes (e.g., primary eclipses in binary systems).
                if abs(z) > self.z_threshold:
                    z_alerts.append(i)

                # Pre-compute relative variance for all sources with enough history
                # (used in Pass 2 below for the pulsator check)
                if len(history) >= MIN_PULSATOR_FRAMES and abs(b_mean) > 1.0:
                    relative_vars[i] = np.var(history) / (b_mean ** 2)

            else:
                z = 0.0
                stds.append(0.0)

            z_scores.append(z)

        # --- Pass 2: Pulsator (Excess Variance) Check ---
        inflations = {}
        for i, flux in enumerate(corrected_fluxes):
            history = self.light_curves[i]
            if len(history) >= MIN_PULSATOR_FRAMES:
                b_mean, b_std = self.baselines[i]
                
                # We skip extremely faint/noisy sources to avoid division instability.
                # Diagnostic tests proved that sources with SNR < 20 have highly
                # non-linear variance (extracting mostly background noise), causing
                # mathematical explosions. True pulsator detection requires clean photometry.
                if b_std > 0 and (abs(b_mean) / b_std) > 20.0:
                    measured_var = np.var(history)
                    baseline_var = b_std ** 2
                    
                    # Inflation factor: how much has this star's variance grown 
                    # compared to its quiet 10-frame baseline?
                    inflations[i] = measured_var / baseline_var
                    
        if inflations:
            # The atmosphere changing over 45 mins causes ALL stars to inflate.
            # We find the median inflation of the field to establish the atmospheric baseline.
            inf_vals = np.array(list(inflations.values()))
            median_inf = np.median(inf_vals)
            mad_inf = np.median(np.abs(inf_vals - median_inf))
            robust_sigma = 1.4826 * mad_inf
            
            if robust_sigma <= 0:
                robust_sigma = 1e-10
            
            # Floor robust_sigma to prevent microscopic atmospheric noise from triggering alerts
            # 0.5 * 5.0 multiplier = minimum 2.5x variance inflation required
            robust_sigma = max(robust_sigma, 0.5)
                
            # Only flag stars that have inflated significantly MORE than the atmosphere caused
            threshold = median_inf + self.var_threshold_multiplier * robust_sigma
            
            for i, inf in inflations.items():
                if inf > threshold:
                    if i not in self.already_alerted_var:
                        var_alerts.append(i)
                        self.already_alerted_var.add(i)

        return np.array(z_scores), np.array(stds), z_alerts, var_alerts

# ============================================================
# interactive_viewer.py
# ============================================================

# -----------------------------------------------------------------------
# LiveDashboard
# A real-time Matplotlib dashboard that connects to a running Orchestrator
# and polls its internal state at a fixed interval.  The heavy lifting
# (calibration, alignment, photometry, alert-logging) all happens in the
# background daemon thread; this class only reads state and renders it.
# -----------------------------------------------------------------------
class LiveDashboard:
    # How often (ms) to poll the Orchestrator for new data
    POLL_INTERVAL_MS = 1500

    def __init__(self, orchestrator):
        """
        Args:
            orchestrator: A fully constructed Orchestrator instance whose
                          watchdog has been started on a daemon thread.
        """
        self.orc = orchestrator

        # Internal display state
        self._last_frame_id    = None   # track when orchestrator pushes a new frame
        self._frame_counter    = 0      # monotonically increasing; survives slews
        self._selected_star_id = None   # index into background_stars_xy
        self._wcs_ready        = False
        self._wcs_timer        = None
        self._transient_coords = []     # (x, y) pairs from discoveries.csv, refreshed live
        
        # Time Travel State
        self._history_cache    = collections.deque(maxlen=100)
        self._is_paused        = False
        self._history_offset   = 0      # 0 is live, -1 is 1 frame back, etc.
        self._last_alert_set   = set()  # track alert changes to force redraws

        self._build_layout()
        self._bind_events()

        # FuncAnimation drives the refresh loop from the main thread —
        # no manual plt.pause() or thread.join() needed.
        self._anim = FuncAnimation(
            self.fig,
            self._tick,
            interval=self.POLL_INTERVAL_MS,
            cache_frame_data=False
        )

    # ------------------------------------------------------------------
    # GUI construction
    # ------------------------------------------------------------------
    def _build_layout(self):
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(11, 13))
        self.fig.patch.set_facecolor('#0d0d0d')
        self.fig.suptitle(
            "MIRA Pipeline  —  Live Dashboard",
            color='white', fontsize=14, fontweight='bold', y=0.98
        )

        # Three rows: image (large) | slim info bar | light-curve
        gs = self.fig.add_gridspec(
            3, 1,
            height_ratios=[6, 0.45, 3],
            hspace=0.08,
            left=0.06, right=0.97, top=0.96, bottom=0.04
        )

        # ---- Top: star-field image ----
        self.ax_img = self.fig.add_subplot(gs[0])
        self.ax_img.set_facecolor('#0d0d0d')
        self.ax_img.set_xticks([])
        self.ax_img.set_yticks([])
        self._img_handle  = None
        self._circle_patches = []

        self._title = self.ax_img.set_title(
            "Waiting for first frame…",
            color='#aaaaaa', fontsize=11, loc='left', pad=6
        )

        # ---- Middle: info bar ----
        self.ax_info = self.fig.add_subplot(gs[1])
        self.ax_info.set_facecolor('#1a1a2e')
        self.ax_info.set_xticks([])
        self.ax_info.set_yticks([])
        for spine in self.ax_info.spines.values():
            spine.set_edgecolor('#333355')

        self._info_text = self.ax_info.text(
            0.5, 0.5,
            "Click a star on the image above to inspect it.",
            ha='center', va='center', fontsize=10,
            color='#888899', transform=self.ax_info.transAxes
        )

        # Live-status badge (top-right corner of info bar)
        self._status_badge = self.ax_info.text(
            0.99, 0.5, "● IDLE",
            ha='right', va='center', fontsize=9,
            color='#555566', transform=self.ax_info.transAxes
        )

        # ---- Bottom: light curve ----
        self.ax_lc = self.fig.add_subplot(gs[2])
        self.ax_lc.set_facecolor('#111122')
        self.ax_lc.set_title("Light Curve", color='#aaaaaa', fontsize=10)
        self.ax_lc.set_xlabel("Frame  #", color='#666677', fontsize=9)
        self.ax_lc.set_ylabel("Relative Flux", color='#666677', fontsize=9)
        self.ax_lc.tick_params(colors='#555566', labelsize=8)
        for spine in self.ax_lc.spines.values():
            spine.set_edgecolor('#333355')
        self._lc_placeholder = self.ax_lc.text(
            0.5, 0.5, "No star selected yet.",
            ha='center', va='center', fontsize=10,
            color='#555566', transform=self.ax_lc.transAxes
        )

    # ------------------------------------------------------------------
    # Event binding
    # ------------------------------------------------------------------
    def _bind_events(self):
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
    def _on_key_press(self, event):
        if not self._history_cache:
            return
            
        if event.key == ' ':
            self._is_paused = not self._is_paused
            if self._is_paused:
                self._title.set_color('#ffaa00')
            else:
                # Snap back to live
                self._history_offset = 0
                self._title.set_color('#aaaaaa')
                
        elif event.key == 'left':
            self._is_paused = True
            self._title.set_color('#ffaa00')
            self._history_offset = max(-len(self._history_cache) + 1, self._history_offset - 1)
            
        elif event.key == 'right':
            self._is_paused = True
            self._title.set_color('#ffaa00')
            self._history_offset = min(0, self._history_offset + 1)
            
        elif event.key == 'home':
            self._is_paused = False
            self._history_offset = 0
            self._title.set_color('#aaaaaa')
            
        # Force an immediate redraw of the selected history frame
        idx = -1 + self._history_offset
        hist_img, hist_count = self._history_cache[idx]
        self._refresh_image(hist_img, self.orc, hist_count)

    # ------------------------------------------------------------------
    # FuncAnimation tick — runs on the main thread every POLL_INTERVAL_MS
    # ------------------------------------------------------------------
    def _tick(self, frame_number):
        """Called by FuncAnimation. Reads the orchestrator's shared state
        and redraws only if something has actually changed."""
        orc = self.orc

        # Nothing rendered yet — orchestrator is still in burn-in
        if orc.reference_image is None or len(orc.background_stars_xy) == 0:
            self._set_status("● BURN-IN", '#e8b84b')
            return

        # Use the most recently aligned image the orchestrator processed.
        # We snapshot it under a lock to avoid tearing.
        current_img = orc._last_aligned_image if getattr(orc, '_last_aligned_image', None) is not None else orc.reference_image
        frame_id    = id(current_img)

        # Always update the underlying history cache when the orchestrator produces a new frame
        if frame_id != self._last_frame_id:
            self._last_frame_id = frame_id
            self._frame_counter += 1
            self._history_cache.append((current_img, self._frame_counter))
            
            # If we are live (not paused and looking at the newest frame), auto-redraw
            if not self._is_paused and self._history_offset == 0:
                self._refresh_image(current_img, orc, self._frame_counter)

        # Refresh alert coords from the live CSV every tick
        prev_alert_count = len(self._transient_coords)
        self._load_alert_coords()
        
        # If new alerts arrived, force an immediate circle redraw even if the image frame
        # hasn't changed — otherwise circles only appear on the NEXT camera frame
        new_alert_set = set(self._transient_coords)
        if new_alert_set != self._last_alert_set:
            self._last_alert_set = new_alert_set
            # Redraw using whatever image is currently displayed
            display_img = current_img if not self._is_paused else self._history_cache[-1 + self._history_offset][0]
            self._refresh_image(display_img, orc, self._frame_counter)

        # Update the status badge
        state_label = orc.state
        if state_label == "MONITORING":
            self._set_status("● MONITORING", '#44ff88')
        else:
            self._set_status(f"● {state_label}", '#e8b84b')

    # ------------------------------------------------------------------
    # Image panel refresh
    # ------------------------------------------------------------------
    def _refresh_image(self, img, orc, render_frame_count):
        """Redraws the star-field and all circle overlays."""
        # Remove old circle patches
        for p in self._circle_patches:
            p.remove()
        self._circle_patches.clear()

        # ZScale stretch
        zscale = ZScaleInterval()
        try:
            vmin, vmax = zscale.get_limits(img)
        except Exception:
            vmin, vmax = np.nanmin(img), np.nanmax(img)

        if self._img_handle is None:
            self._img_handle = self.ax_img.imshow(
                img, cmap='gray', origin='lower', vmin=vmin, vmax=vmax,
                interpolation='nearest', aspect='auto'
            )
        else:
            self._img_handle.set_data(img)
            self._img_handle.set_clim(vmin, vmax)

        # Alert set for fast lookup
        alert_set = set(self._transient_coords)

        for i, (x, y) in enumerate(orc.background_stars_xy):
            is_transient = any(
                abs(x - tx) < 8 and abs(y - ty) < 8
                for (tx, ty) in alert_set
            )
            
            if i == self._selected_star_id:
                color = '#44ff88' # Green for selected
                lw = 2.5
                radius = 14
            else:
                color  = '#ff4444' if is_transient else '#4488ff'
                lw     = 2.0       if is_transient else 1.2
                radius = 12        if is_transient else 10
            circ = patches.Circle(
                (x, y), radius=radius,
                edgecolor=color, facecolor='none',
                lw=lw, alpha=0.75
            )
            self.ax_img.add_patch(circ)
            self._circle_patches.append(circ)
            
        # Engine A alerts are NEW transients (not in background_stars_xy)
        # So we must draw them explicitly!
        for (tx, ty) in alert_set:
            # Only draw if we haven't already drawn it as a background star
            already_drawn = any(
                abs(x - tx) < 8 and abs(y - ty) < 8
                for (x, y) in orc.background_stars_xy
            )
            if not already_drawn:
                circ = patches.Circle(
                    (tx, ty), radius=12,
                    edgecolor='#ff4444', facecolor='none',
                    lw=2.0, alpha=0.75
                )
                self.ax_img.add_patch(circ)
                self._circle_patches.append(circ)

        # Update frame counter title
        n_alerts = len(alert_set)
        n_stars  = len(orc.background_stars_xy)
        
        status_prefix = ""
        if self._is_paused or self._history_offset < 0:
            status_prefix = f"[PAUSED - History: {self._history_offset}]  "
            
        self._title.set_text(
            f"{status_prefix}Frame {render_frame_count}  |  Stars tracked: {n_stars}  |  Alerts: {n_alerts}"
        )

        # Re-plot light curve for the selected star (new frame appended)
        if self._selected_star_id is not None:
            self._plot_lightcurve(self._selected_star_id, render_frame_count - 1)

    # ------------------------------------------------------------------
    # Light-curve panel
    # ------------------------------------------------------------------
    def _plot_lightcurve(self, star_idx, current_frame_idx):
        lc_data = self.orc.photometry_engine.light_curves.get(star_idx, [])
        self.ax_lc.clear()
        self.ax_lc.set_facecolor('#111122')
        self.ax_lc.set_title("Light Curve", color='#aaaaaa', fontsize=10)
        self.ax_lc.set_xlabel("Frame  #", color='#666677', fontsize=9)
        self.ax_lc.set_ylabel("Relative Flux", color='#666677', fontsize=9)
        self.ax_lc.tick_params(colors='#555566', labelsize=8)
        for spine in self.ax_lc.spines.values():
            spine.set_edgecolor('#333355')

        if not lc_data:
            self.ax_lc.text(
                0.5, 0.5, "No photometry data yet.",
                ha='center', va='center', fontsize=10,
                color='#555566', transform=self.ax_lc.transAxes
            )
            return

        frames = list(range(len(lc_data)))
        
        # Normalize to relative flux (~1.0)
        median_flux = np.median(lc_data)
        if median_flux > 0:
            norm_lc = [f / median_flux for f in lc_data]
        else:
            norm_lc = lc_data
            
        self.ax_lc.plot(frames, norm_lc, color='#4488ff', lw=1.5, alpha=0.9)
        self.ax_lc.fill_between(frames, norm_lc, alpha=0.08, color='#4488ff')

        # Highlight current live frame
        if 0 <= current_frame_idx < len(norm_lc):
            self.ax_lc.axvline(current_frame_idx, color='#ff4444', lw=1.2, linestyle='--', alpha=0.7)
            self.ax_lc.scatter(
                [current_frame_idx], [norm_lc[current_frame_idx]],
                color='#ff4444', zorder=5, s=40
            )

    # ------------------------------------------------------------------
    # Click handler
    # ------------------------------------------------------------------
    def _on_click(self, event):
        if event.inaxes != self.ax_img:
            return

        cx, cy = event.xdata, event.ydata
        if cx is None or cy is None:
            return

        stars = self.orc.background_stars_xy
        if not stars:
            return

        dists = [(px - cx) ** 2 + (py - cy) ** 2 for (px, py) in stars]
        min_idx  = int(np.argmin(dists))
        min_dist = dists[min_idx]

        if min_dist > 625:  # 25-pixel radius click tolerance
            return

        self._selected_star_id = min_idx
        star_x, star_y = stars[min_idx]

        wcs = self.orc.current_wcs
        
        # Base string is always X/Y pixel coordinates
        coord_text = f"*  Star #{min_idx}  |  X: {star_x:.1f}  Y: {star_y:.1f}"

        if wcs is not None:
            try:
                sky = wcs.pixel_to_world(star_x, star_y)
                coord_text += f"  |  RA: {sky.ra.deg:.5f}°  Dec: {sky.dec.deg:.5f}°"
            except Exception:
                coord_text += "  |  RA/Dec: Error"
        else:
            coord_text += "  |  has not finished rendering"

        self._info_text.set_text(coord_text)
        self._info_text.set_color('#44ff88')

        self._plot_lightcurve(min_idx, self._frame_counter - 1)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _set_status(self, label, color):
        self._status_badge.set_text(label)
        self._status_badge.set_color(color)

    def _load_alert_coords(self):
        alerts_path = os.path.join(
            getattr(self.orc.alert_logger, 'output_dir', 'pipeline_discoveries'),
            'discoveries.csv'
        )
        if not os.path.exists(alerts_path):
            self._transient_coords = []
            return
        coords = []
        try:
            with open(alerts_path, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                if headers is None:
                    return
                # Find X_Pixel and Y_Pixel columns by name (robust against column reordering)
                try:
                    x_idx = headers.index('X_Pixel')
                    y_idx = headers.index('Y_Pixel')
                except ValueError:
                    return  # Headerless or corrupt file — skip silently
                for row in reader:
                    try:
                        if len(row) > max(x_idx, y_idx):
                            coords.append((float(row[x_idx]), float(row[y_idx])))
                    except (ValueError, IndexError):
                        pass
            # Only replace coords on a clean successful read
            self._transient_coords = coords
        except Exception:
            pass  # Keep old coords if the file is temporarily locked

    def show(self):
        """Hand control to the Matplotlib event loop (must be called from main thread)."""
        plt.show()

# ============================================================
# src/orchestrator.py
# ============================================================

# Import our custom modules

# Set up logging for the daemon so output isn't lost
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("mira_pipeline.log"),
        logging.StreamHandler()
    ]
)

class Orchestrator:
    """
    The Master Hardware Loop.
    Because the camera cannot talk to the telescope or the pipeline, this script acts
    as an autonomous daemon. It continuously polls a spool folder for new FITS images,
    pushes them through the engines, and gracefully handles telescope slewing.
    """
    def __init__(self, spool_directory="camera_spool", search_pattern="*fit*", flat=None, bias=None, hot_pixels_path=None):
        self.spool_directory = spool_directory
        self.search_pattern = search_pattern
       
        # Memory Leak Fix: Use a deque with a maxlen instead of an infinitely growing set
        self.processed_files = deque(maxlen=2000)
       
        # Hardware calibration masters (Mocked as 0/1 for fallback)
        self.flat = flat if flat is not None else np.ones((1, 1))
        self.bias = bias if bias is not None else np.zeros((1, 1))
       
        # Load Bad Pixel Mask dynamically
        if hot_pixels_path and os.path.exists(hot_pixels_path):
            try:
                dark_data = fits.getdata(hot_pixels_path)
                mean_val = np.mean(dark_data)
                std_val = np.std(dark_data)
                threshold = mean_val + (3 * std_val)
               
                # Mask anything > 3 standard deviations above the mean dark current
                self.bad_pixel_mask = dark_data > threshold
                logging.info(f"Loaded Physical Bad Pixel Mask: {np.sum(self.bad_pixel_mask)} pixels flagged (>3 sigma).")
            except Exception as e:
                logging.error(f"Failed to load hot pixels mask: {e}")
                self.bad_pixel_mask = None
        else:
            logging.info("No bad pixel mask found. Proceeding without one.")
            self.bad_pixel_mask = None
           
        # Pipeline State
        self.state = "BURN_IN"
        self.burn_in_cache = []
        self.reference_image = None
        self.background_stars_xy = []
        self.current_wcs = None
        self.alignment_failures = 0
       
        # Initialize the Engines
        self.photometry_engine = PhotometryEngine()
        self.temporal_verifier = TemporalVerifier(required_consecutive=3)
        self.alert_logger = AlertLogger()

        # Shared state read by the LiveDashboard on the main thread.
        # Written only inside process_new_image() on the daemon thread.
        self._last_aligned_image = None

    def reset_pipeline(self, reason):
        """Called when astroalign detects the telescope has moved."""
        logging.warning(f"[SYSTEM RESET] {reason}")
        logging.info("Flushing cache and initiating new Burn-In Phase...")
        self.state = "BURN_IN"
        self.burn_in_cache = []
        self.reference_image = None
        self.background_stars_xy = []
        self.current_wcs = None
        self.photometry_engine = PhotometryEngine() # Reset flux history
        self.temporal_verifier = TemporalVerifier() # Reset temporal history
        self.alert_logger.rotate_csv() # Archive old discoveries and start fresh

    def _async_solve_wcs(self, filepath):
        """Runs the astrometry solver in a background thread to prevent blocking the daemon."""
        logging.info("Starting background WCS solver...")
        wcs_result = solve_wcs_for_image(filepath)
        if wcs_result is not None:
            self.current_wcs = wcs_result
            logging.info("Background WCS lock acquired.")
        else:
            logging.warning("Background WCS solver failed.")

    def process_new_image(self, filepath):
        """Passes a single new image through the pipeline architecture."""
        logging.info(f"Processing: {os.path.basename(filepath)}")
       
        try:
            with fits.open(filepath) as hdul:
                raw_image = hdul[0].data
        except Exception as e:
            logging.error(f"Failed to read FITS file. Skipping. Error: {e}")
            return

        # Resize mock calibration frames if necessary to match data
        if self.flat.shape != raw_image.shape:
            self.flat = np.ones_like(raw_image, dtype=float)
            self.bias = np.zeros_like(raw_image, dtype=float)
           
        clean_image = calibrate_image(raw_image, self.bias, self.flat, bad_pixel_mask=self.bad_pixel_mask)

        # ---------------------------------------------------------
        # PHASE 1: BURN-IN
        # ---------------------------------------------------------
        if self.state == "BURN_IN":
            self.burn_in_cache.append(clean_image)
            logging.info(f"Burn-In Phase: Frame {len(self.burn_in_cache)}/5 collected.")
           
            if len(self.burn_in_cache) == 5:
                logging.info("Burn-In Complete! Generating Dynamic Reference...")
                try:
                    # File 1: Generate Master Reference
                    self.reference_image = generate_master_reference(self.burn_in_cache)
                   
                    # File 2: Autonomously map the stars
                    self.background_stars_xy = find_stars_autonomously(self.reference_image)
                    logging.info(f"StarFinder locked onto {len(self.background_stars_xy)} background stars.")
                   
                    # File 3: Attempt to generate World Coordinate System asynchronously
                    # This prevents the cloud API timeout from freezing the pipeline
                    threading.Thread(target=self._async_solve_wcs, args=(filepath,), daemon=True).start()
                   
                    self.state = "MONITORING"
                except AlignmentError as e:
                    self.reset_pipeline(f"Telescope slewed during Burn-In: {e}")
            return # Wait for next frame

        # ---------------------------------------------------------
        # PHASE 2: CONTINUOUS MONITORING (Engines A & B)
        # ---------------------------------------------------------
        try:
            # First, align the current frame to the reference
            aligned_image = align_image(clean_image, self.reference_image)
            self.alignment_failures = 0  # Reset on success
        except AlignmentError as e:
            self.alignment_failures += 1
            if self.alignment_failures >= 3:
                # THE HARDWARE TRIGGER: The telescope moved!
                self.reset_pipeline(f"Telescope Slew Confirmed! ({e})")
                self.burn_in_cache.append(clean_image) # Use this frame as frame 1 of new burn-in
            else:
                logging.warning(f"Astroalign failed (Strike {self.alignment_failures}/3). Likely a cloud or satellite trail. Dropping frame.")
            return

        # Expose the latest aligned frame to the LiveDashboard.
        # This is a simple assignment — safe to read from the main thread
        # because Python's GIL ensures object-reference updates are atomic.
        self._last_aligned_image = aligned_image

        raw_candidates = [] # Stores dicts: x, y, engine, sig, bypass_bouncer

        # -- ENGINE B (Photometry) --
        fluxes = self.photometry_engine.perform_aperture_photometry(aligned_image, self.background_stars_xy)
        z_scores, stds, z_alerts, var_alerts = self.photometry_engine.update_light_curves(fluxes)
       
        # Flares are 1-frame events. They bypass the temporal bouncer.
        for idx in z_alerts:
            x, y = self.background_stars_xy[idx]
            sig = f"Z={z_scores[idx]:.1f}"
            raw_candidates.append({'x': x, 'y': y, 'engine': 'Engine B (Flare)', 'sig': sig, 'bypass': True})
           
        # Pulsators are slow variables. They go to the bouncer.
        for idx in var_alerts:
            x, y = self.background_stars_xy[idx]
            sig = f"Var={stds[idx]:.1f}"
            raw_candidates.append({'x': x, 'y': y, 'engine': 'Engine B (Pulsator)', 'sig': sig, 'bypass': False})

        # -- ENGINE A (Optimal Image Subtraction) --
        diff_image = optimal_image_subtraction(aligned_image, self.reference_image)
        new_objects, bkg_rms = extract_sources_from_difference(diff_image)
        for obj in new_objects:
            if spatial_profile_vetting(obj):
                # Saturation Check: Ensure this isn't a blooming artifact from a bright star
                if saturation_vetting(obj['x'], obj['y'], aligned_image):
                    # CROSS-MATCH VETO: Is this "new" object actually just a poorly subtracted known star?
                    # Engine A is for discovering uncataloged objects in empty space. 
                    # If it's a known star, Engine B handles it.
                    is_known = any(abs(obj['x'] - bx) < 3.0 and abs(obj['y'] - by) < 3.0 for bx, by in self.background_stars_xy)
                    if not is_known:
                        sigma = obj['peak'] / bkg_rms if bkg_rms > 0 else 0
                        sig = f"Sigma={sigma:.1f}"
                        raw_candidates.append({'x': obj['x'], 'y': obj['y'], 'engine': 'Engine A (New)', 'sig': sig, 'bypass': False})
                   
        # -- MERGE CO-DETECTED TRANSIENTS --
        # If an object triggers BOTH Engine A and Engine B, merge them into a single alert
        merged_candidates = []
        for det in raw_candidates:
            matched = False
            for m in merged_candidates:
                dist = np.sqrt((det['x'] - m['x'])**2 + (det['y'] - m['y'])**2)
                if dist < 3.0: # within 3 pixels
                    if det['engine'] not in m['engine']:
                        m['engine'] += f" + {det['engine']}"
                        m['sig'] += f" | {det['sig']}"
                    m['bypass'] = m['bypass'] or det['bypass']
                    matched = True
                    break
            if not matched:
                merged_candidates.append(det)

        # -- ATMOSPHERIC DEGRADATION GATE --
        # If a cloud causes hundreds of stars to suddenly fluctuate, drop the frame.
        if len(merged_candidates) > 100:
            logging.warning(f"Atmospheric Degradation Detected! {len(merged_candidates)} simultaneous alerts. Dropping frame.")
            return # Exit and wait for the next frame

        # ---------------------------------------------------------
        # PHASE 3: TEMPORAL VETTING & LOGGING
        # ---------------------------------------------------------
        bouncer_candidates = []
       
        for m in merged_candidates:
            if m['bypass']:
                # Log immediately, bypass temporal verification
                self.alert_logger.log_alert(m['engine'], m['x'], m['y'], aligned_image, wcs=self.current_wcs, significance=m['sig'])
            else:
                bouncer_candidates.append(m)
               
        # To track objects temporally, we pass their raw float X,Y coordinates
        coord_ids = [(float(m['x']), float(m['y'])) for m in bouncer_candidates]
       
        survivors = self.temporal_verifier.verify(coord_ids)
       
        for survivor_id in survivors:
            # Find the original candidate data to pass to the logger
            for m in bouncer_candidates:
                if (float(m['x']), float(m['y'])) == survivor_id:
                    self.alert_logger.log_alert(m['engine'], m['x'], m['y'], aligned_image, wcs=self.current_wcs, significance=m['sig'])
                    break # Logged

    def run_watchdog(self):
        """The infinite polling loop designed for an air-gapped machine."""
        logging.info(f"Starting Orchestrator. Watching directory: {self.spool_directory}")
        if not os.path.exists(self.spool_directory):
            os.makedirs(self.spool_directory)
           
        while True:
            # Find all fits files matching the pattern, sorted alphabetically by filename
            fits_files = sorted(glob.glob(os.path.join(self.spool_directory, self.search_pattern)))
           
            for filepath in fits_files:
                if filepath not in self.processed_files:
                    # I/O RACE CONDITION PATCH: File Stability Lock
                    # Ensure the camera has finished writing the file to disk before reading it
                    size_1 = os.path.getsize(filepath)
                    time.sleep(0.5)
                    size_2 = os.path.getsize(filepath)
                   
                    if size_1 == size_2 and size_1 > 0:
                        self.process_new_image(filepath)
                        self.processed_files.append(filepath)
                   
            # Wait before checking the folder again
            time.sleep(2)

    def start_live_dashboard(self, spool_dir=None):
        """
        Entry point for Option A — Live Dashboard mode.

        Spawns the watchdog loop on a background daemon thread so the camera
        spool is processed continuously, then hands the main thread to
        Matplotlib's event loop so the GUI stays responsive.
        """
        # Import here to avoid circular deps and to keep orchestrator
        # importable on headless servers without matplotlib installed.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        if spool_dir:
            self.spool_directory = spool_dir

        logging.info("[DASHBOARD] Launching Live Dashboard — watchdog starting on daemon thread.")

        # Daemon thread: dies automatically when the main (GUI) thread exits
        watchdog_thread = threading.Thread(
            target=self.run_watchdog,
            name="mira-watchdog",
            daemon=True
        )
        watchdog_thread.start()

        # Main thread: build + run the GUI
        dashboard = LiveDashboard(orchestrator=self)
        dashboard.show()   # blocks until the window is closed


# ============================================================
# EXECUTION LOGIC
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Launch the MIRA Pipeline GUI.")
    parser.add_argument("--pattern", type=str, default="*g*fit*", 
                        help="Glob pattern to enforce tracking the same star and filter.")
    parser.add_argument("--filter", type=str, default="g", choices=['g', 'i', 'r'],
                        help="Which filter to load calibration frames for (g, i, or r).")
    args = parser.parse_args()

    print("Initializing MIRA Pipeline GUI...")
    
    ssd_root = get_extreme_ssd_path()
    if not ssd_root:
        print("WARNING: 'Extreme SSD' not found. Falling back to current directory.")
        ssd_root = os.getcwd()
    else:
        print(f"SUCCESS: 'Extreme SSD' mounted at {ssd_root}")
        
    hot_pixels_path = os.path.join(ssd_root, 'hot_pixels.fts')
    
    # --- LOAD CALIBRATION FRAMES ---
    calib_dir = os.path.join(ssd_root, 'calibration frames')
    flat_data, bias_data = None, None

    # Try .fits extension first, fall back to .fts
    def _find_calib_file(calib_dir, basename):
        for ext in ['.fits', '.fts', '.fit']:
            p = os.path.join(calib_dir, basename + ext)
            if os.path.exists(p):
                return p
        return None

    bias_path = _find_calib_file(calib_dir, 'median_bias')
    flat_path = _find_calib_file(calib_dir, f'mean_flat_{args.filter}')

    try:
        if bias_path:
            with fits.open(bias_path) as hdul:
                bias_data = hdul[0].data.astype(float)
            print(f"Loaded master bias: {bias_path}")
        else:
            print(f"Warning: Master bias not found in '{calib_dir}' (tried .fits/.fts/.fit)")

        if flat_path:
            with fits.open(flat_path) as hdul:
                flat_data = hdul[0].data.astype(float)
            print(f"Loaded master flat ({args.filter}): {flat_path}")
        else:
            print(f"Warning: Master flat not found in '{calib_dir}' (tried .fits/.fts/.fit)")
    except Exception as e:
        print(f"Error loading calibration frames: {e}")

    try:
        folder_name = input("Enter the folder name for tonight's data (e.g. 20260722): ").strip()
    except EOFError:
        folder_name = "test_data"

    # Images are stored inside Desktop\14-inch Photometry\<folder_name>
    desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
    target_dir = os.path.join(desktop_dir, "14-inch Photometry", folder_name)

    if not os.path.exists(target_dir):
        print(f"Warning: Target directory '{target_dir}' not found. The GUI will still launch and watch this folder for new files.")

    print(f"Watching directory '{target_dir}' for pattern '{args.pattern}'...")
    
    orchestrator = Orchestrator(
        spool_directory=target_dir, 
        search_pattern=args.pattern,
        hot_pixels_path=hot_pixels_path,
        flat=flat_data,
        bias=bias_data
    )
    
    orchestrator.start_live_dashboard()

if __name__ == "__main__":
    main()
