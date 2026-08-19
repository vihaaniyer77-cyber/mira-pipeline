#UPDATED
#UPDATE 2
# VarWatch — MIRA real-time transient detection & photometric monitoring pipeline
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

# [P1] Cap heavy math at (cores - 2): workers=-1 pinned every core at 100%,
# starving the GUI and (on the observatory machine) likely solve-field itself.
FFT_WORKERS = max(1, (os.cpu_count() or 4) - 2)

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
from matplotlib.collections import EllipseCollection
from matplotlib.widgets import Button, RadioButtons

from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ZScaleInterval
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry

# --- DYNAMIC DRIVE DETECTION ---
def get_extreme_ssd_path():
    # Windows-only: volume-label scan for the observatory's 'Extreme SSD'.
    # On macOS/Linux this returns None immediately and callers fall back
    # to CLI-provided or current-directory paths.
    if sys.platform != "win32":
        return None
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

        # [P2] A new session must never display the previous night's alerts:
        # archive any existing discoveries.csv automatically at startup.
        if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 200:
            self.rotate_csv()
        else:
            self._init_csv()

    def _init_csv(self):
        # Initialize the CSV with headers if it doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Obs_UTC", "Obs_JD", "Logged_Local", "Engine", "Filter", "Star_ID", "Significance", "X_Pixel", "Y_Pixel", "RA", "Dec", "RA_HMS", "Dec_DMS", "Image_File"])

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

    def log_alert(self, engine_name, x, y, full_image, crop_size=50, wcs=None, significance="Unknown",
                  filter_name=None, obs_jd=None, obs_datetime=None, star_id=None):

        # Ensure integers for indexing
        x, y = int(round(x)), int(round(y))

        # 1. Generate a unique timestamp filename (processing wall-clock — for
        # filename uniqueness only; scientific times below come from the header)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        engine_short = engine_name.split()[1].replace("(", "").replace(")", "") if " " in engine_name else engine_name
        txt_filename = f"{timestamp}_{engine_short}_X{x}_Y{y}.txt"
        txt_filepath = os.path.join(self.output_dir, txt_filename)

        # [R1] Observation time from the FITS header (UTC), NOT the local
        # wall-clock at processing time — the alert must record when the
        # photons arrived.
        obs_utc = str(obs_datetime) if obs_datetime else "Unknown"
        obs_jd_str = f"{obs_jd:.6f}" if obs_jd is not None else "Unknown"

        # 2. Handle Astrometry (WCS Transformation)
        ra_str = "Unknown"
        dec_str = "Unknown"
        ra_hms = "Unknown"
        dec_dms = "Unknown"
        if wcs is not None:
            try:
                sky_coord = wcs.pixel_to_world(x, y)
                ra_str = f"{sky_coord.ra.deg:.5f}"
                dec_str = f"{sky_coord.dec.deg:.5f}"
                # [P2] Sexagesimal alongside decimal degrees
                ra_hms = sky_coord.ra.to_string(unit='hourangle', sep=':', precision=2, pad=True)
                dec_dms = sky_coord.dec.to_string(sep=':', precision=1, alwayssign=True, pad=True)
            except Exception as e:
                print(f"Warning: Failed to convert pixels to RA/Dec: {e}")

        # [R8] Write the alert text file ALWAYS — previously it was only
        # written when a WCS existed, so the CSV referenced files that were
        # never created whenever astrometry had failed.
        try:
            with open(txt_filepath, 'w') as f:
                f.write(f"--- VarWatch ALERT ---\n")
                f.write(f"Type: {engine_name}\n")
                f.write(f"Filter: {filter_name or '?'}\n")
                f.write(f"Significance: {significance}\n")
                f.write(f"Obs time (UTC): {obs_utc}\n")
                f.write(f"Obs JD: {obs_jd_str}\n")
                f.write(f"RA (deg): {ra_str}\n")
                f.write(f"Dec (deg): {dec_str}\n")
                f.write(f"RA (hms): {ra_hms}\n")
                f.write(f"Dec (dms): {dec_dms}\n")
                f.write(f"Pixel: X:{x}, Y:{y}\n")
        except Exception as e:
            print(f"Warning: Failed to write alert txt: {e}")

        # 5. Append to the localized CSV database
        try:
            with open(self.csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([obs_utc, obs_jd_str, timestamp, engine_name, filter_name or "?",
                                 star_id if star_id is not None else "",
                                 significance, x, y, ra_str, dec_str, ra_hms, dec_dms, txt_filename])
        except Exception as e:
            print(f"Warning: Failed to write to CSV database: {e}")

        # 6. Console Notification
        print(f"DISCOVERY LOGGED [{engine_name} | {filter_name or '?'} | {significance}] at RA:{ra_str}, Dec:{dec_str} (X:{x}, Y:{y}) obs={obs_utc}. Saved to {txt_filename}")

# ============================================================
# src/astrometry_solver.py
# ============================================================

def _solve_wcs_with_astrometry_package(fits_filepath):
    """
    Solver tier 2: the `astrometry` pip package (pip install astrometry) —
    the astrometry.net engine as a Python wheel. Solves from the pipeline's
    own star detections in seconds (measured 2.8 s blind on a KIC frame vs
    minutes for CLI solve-field on the full image).

    NOTE: like all of astrometry.net, this package is Linux/macOS only — no
    Windows wheels. On the observatory Windows machines the WSL solve-field
    path above remains the primary (and correct) solver; this tier exists for
    remote testing on Mac/Linux and simply fails to import on Windows,
    falling through to the cloud API.

    Index files are NOT auto-downloaded (they are ~GB): this tier activates
    only if index-*.fits files already exist in MIRA_ASTROMETRY_INDEX_DIR
    (default ~/.mira_astrometry_index). One-time setup in Python:
        import astrometry
        astrometry.series_4200.index_files(cache_directory='<dir>', scales={3, 4})
    """
    try:
        import astrometry
    except ImportError:
        return None

    index_dir = os.environ.get("MIRA_ASTROMETRY_INDEX_DIR",
                               os.path.join(os.path.expanduser("~"), ".mira_astrometry_index"))
    import pathlib
    index_files = sorted(pathlib.Path(p) for p in
                         glob.glob(os.path.join(index_dir, "**", "index-*.fits"), recursive=True))
    if not index_files:
        print(f"astrometry package present but no index files under {index_dir} — skipping this tier.")
        return None

    try:
        with fits.open(fits_filepath) as hdul:
            image = hdul[0].data.astype(float)
        coords, _fluxes = _detect_sources(image)
        if len(coords) < 10:
            print("astrometry package tier: too few stars detected to solve.")
            return None

        solver = astrometry.Solver(index_files)
        sol = solver.solve(
            stars=coords[:60],
            size_hint=astrometry.SizeHint(lower_arcsec_per_pixel=0.55, upper_arcsec_per_pixel=0.70),
            position_hint=None,
            solution_parameters=astrometry.SolutionParameters(),
        )
        if sol.has_match():
            m = sol.best_match()
            print(f" Astrometry (python package) SUCCESS: RA={m.center_ra_deg:.4f} Dec={m.center_dec_deg:.4f} "
                  f"scale={m.scale_arcsec_per_pixel:.3f}\"/px")
            # wcs_fields values are (value, comment) tuples — must be assigned
            # per-key (fits.Header(dict) treats the tuple itself as the value)
            header = fits.Header()
            for key, value in m.wcs_fields.items():
                header[key] = value
            return WCS(header)
        print(" Astrometry (python package): no match.")
        return None
    except Exception as e:
        print(f" Astrometry (python package) ERROR: {e}")
        return None


def solve_wcs_for_image(fits_filepath):

    if not os.path.exists(fits_filepath):
        print(f"File not found: {fits_filepath}")
        return None
        
    base_path, _ = os.path.splitext(fits_filepath)
    wcs_output_path = base_path + ".wcs"
    #IMPORTANT, MAKE SURE TO DOWNLOAD THE FILES BEFOREHAND!!!!!!
    try:
        # --downsample 2: bins the image before source extraction — standard
        # practice for 15-megapixel frames, typically several-fold faster with
        # no loss of solve reliability.
        solve_args = ["solve-field", "--overwrite", "--no-plots", "--cpulimit", "600",
                      "--downsample", "2",
                      "--scale-units", "arcsecperpix", "--scale-low", "0.59", "--scale-high", "0.62"]

        if sys.platform == "win32":
            # Observatory machine: solve-field lives inside WSL.
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
            cmd = ["wsl"] + solve_args + [linux_filepath]
        else:
            # macOS/Linux: call a native solve-field directly if installed.
            cmd = solve_args + [fits_filepath]

        # Run the local solve-field command
        # --overwrite: Overwrite existing .wcs files
        # --no-plots: We don't need astrometry.net generating annotated images
        # --cpulimit 600: Fail fast if it can't solve in 600 seconds (10 minutes)
        # --scale-units arcsecperpix / --scale-low / --scale-high: Provide telescope scale hints to speed up solving
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Log the solver's full output — a failed solve's cause is almost
        # always in stdout/stderr, and truncating it hides the diagnosis.
        if result.returncode != 0:
            logging.warning(f"solve-field exited {result.returncode}.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        
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
            # Tier 2: python-package engine before giving up
            wcs_pkg = _solve_wcs_with_astrometry_package(fits_filepath)
            if wcs_pkg is not None:
                return wcs_pkg
            print(" Falling back to X/Y pixels.")
            return None

    except FileNotFoundError:
        # solve-field is missing. Tier 2: python-package engine (no WSL/CLI needed)
        wcs_pkg = _solve_wcs_with_astrometry_package(fits_filepath)
        if wcs_pkg is not None:
            return wcs_pkg

        # Tier 3: Astrometry.net Web API via astroquery
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

def single_pixel_vetting(x, y, image, search_box=4, dominance_ratio=3.0):
    """
    Rejects detections whose flux lives in ONE dominant pixel — hot/warm
    pixels and cosmic-ray hits — by inspecting the UNCONVOLVED aligned frame.

    Why this is needed despite the FWHM check: the subtraction kernel can
    smear a single hot pixel into a PSF-shaped blob on the difference image
    (when the 'blur target' direction is chosen), so shape vetting on the
    difference passes it. A real star at 3-5 px seeing puts at most ~40% of
    its peak in one pixel; a defect puts ~everything there.

    Warm pixels are also invisible to every other guard: they are absent from
    the median reference (alignment shifts move them between frames, the
    median suppresses them), so there is no veto-list entry and no negative
    dipole lobe — and they repeat every frame, so temporal verification
    happily confirms them. Measured on 20260509: two warm pixels produced
    persistent 193-sigma and 686-sigma 'discoveries'.

    Returns True if the detection looks like a real (extended) source.
    """
    x, y = int(round(x)), int(round(y))
    y0, y1 = max(1, y - search_box), min(image.shape[0] - 1, y + search_box + 1)
    x0, x1 = max(1, x - search_box), min(image.shape[1] - 1, x + search_box + 1)
    patch = image[y0:y1, x0:x1].astype(float)
    if patch.size < 9:
        return False
    # Local background from a wider ring
    wy0, wy1 = max(0, y - 15), min(image.shape[0], y + 16)
    wx0, wx1 = max(0, x - 15), min(image.shape[1], x + 16)
    bkg_med = np.median(image[wy0:wy1, wx0:wx1])

    # Locate the brightest pixel near the detection, then compare it to its
    # 8 IMMEDIATE neighbors only. (Comparing against the whole patch fails
    # when an unrelated star's wing supplies a bright 'second' pixel.)
    iy, ix = np.unravel_index(np.argmax(patch), patch.shape)
    py, px = y0 + iy, x0 + ix
    peak = image[py, px] - bkg_med
    if peak <= 0:
        return False
    neigh = image[py-1:py+2, px-1:px+2].astype(float) - bkg_med
    neigh_max = np.partition(neigh.ravel(), -2)[-2]  # brightest excluding center
    # A defect towers over its brightest neighbor; a star (or any
    # seeing-limited source) does not.
    return peak < dominance_ratio * max(neigh_max, 1e-6)


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
                # BUG FIX: the 'alerted' flag must only be consumed when the
                # alert is actually DELIVERED. Previously it was set here even
                # during the baseline quiet period (whose gate below discards
                # valid_targets), so any transient reaching 3 consecutive
                # detections within the first baseline_frames was silently
                # swallowed forever. Found by synthetic-SN injection testing.
                if self.history[best_match_id]['count'] >= self.required:
                    if not self.history[best_match_id]['alerted'] and self.frame_count > self.baseline_frames:
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

def _detect_sources(image, fwhm_estimate=3.0, threshold_sigma=5.0, saturation_level=55000.0):
    """Runs DAOStarFinder on a locally-noise-normalized (SNR) image and returns
    (coords Nx2 array, fluxes N array), brightest first.

    Detection happens in SNR space — (image - local background) / local RMS —
    rather than against a single global sigma. Flat-field correction amplifies
    fixed-pattern noise toward the frame edges, so a global threshold floods
    the star list with edge artifacts while staying blind to real stars in the
    quiet center. Local normalization makes '5 sigma' mean the same thing
    everywhere on the frame."""
    img_data = np.ascontiguousarray(image, dtype=np.float64)
    bkg = sep.Background(img_data)
    rms = bkg.rms()
    # Guard dead/zero-noise regions
    rms = np.maximum(rms, 1e-3)
    snr_image = (img_data - bkg.back()) / rms

    # https://photutils.readthedocs.io/en/stable/user_guide/index.html?__cf_chl_f_tk=nCEvkYnwI47vL8uPS2VDGbWkpUYCSbQ13nkrgW8C240-1783358576-1.0.1.1-Ioz4b5Z.o94sDQmV5ijRN.kaBAfrbOX_MRopAMHCuKU
    daofind = DAOStarFinder(fwhm=fwhm_estimate, threshold=threshold_sigma, sharplo=0.2, sharphi=0.8)

    sources = daofind(snr_image)

    if sources is None or len(sources) == 0:
        return np.empty((0, 2)), np.empty(0)

    coords = np.array([(row['xcentroid'], row['ycentroid']) for row in sources])
    fluxes = np.array([row['flux'] for row in sources])

    # Saturation cut in real ADU (peakmax can't apply in SNR space)
    ix = np.clip(coords[:, 0].round().astype(int), 0, image.shape[1] - 1)
    iy = np.clip(coords[:, 1].round().astype(int), 0, image.shape[0] - 1)
    unsaturated = image[iy, ix] < saturation_level
    coords, fluxes = coords[unsaturated], fluxes[unsaturated]

    # Sort by brightest (highest SNR) first
    order = np.argsort(fluxes)[::-1]
    return coords[order], fluxes[order]


def select_persistent_stars(frames, reference_image=None, min_frac=0.6, tolerance=1.5,
                            max_stars=500, fwhm_estimate=3.0, threshold_sigma=5.0,
                            confirm_sigma=3.0, saturation_level=55000.0, min_separation=16.0,
                            valid_mask=None, edge_margin=60, return_all=False, saturation_ref=None):
    """
    Selects real stars: deep detection on the stacked reference, then
    cross-frame persistence confirmation on the individual burn-in frames.

    On short exposures the single-frame 5-sigma detection list is dominated by
    noise peaks (measured on the 20251205 epsPer data: ~4,400 detections of
    which ~113 are real). Two-stage selection fixes both halves of the problem:

    1. CANDIDATES come from the median-stacked reference image (sqrt(N) deeper
       than any single frame), so faint real stars that hover at the single-
       frame threshold are found reliably.
    2. CONFIRMATION requires each candidate to be re-detected in >= min_frac
       of the individual frames within `tolerance` px — at a relaxed
       confirm_sigma, because confirmation of a known position needs less
       significance than blind detection. Noise peaks in the stack do not
       repeat in individual frames and are rejected.

    Crowding: for pairs closer than min_separation the FAINTER member is
    dropped (dropping both eliminated the brightest star in the field via its
    own PSF-wing deblends).

    Args:
        frames: list of 2D arrays (the burn-in frames, ideally aligned).
        reference_image: the median stack of `frames` (falls back to frames[0]).
    Returns:
        list of (x, y) tuples, brightest first, capped at max_stars.
    """
    if reference_image is None:
        reference_image = frames[0]

    # Stage 1: deep candidate detection on the stack
    cand_xy, cand_flux = _detect_sources(reference_image, fwhm_estimate,
                                         threshold_sigma, saturation_level)
    if len(cand_xy) == 0:
        return []

    # Reject candidates near SATURATED cores. Saturated stars produce garbage
    # photometry (measured: fake 4-15% 'variability' on railed stars), and
    # their wings contaminate neighbors — so exclude a 40 px zone around any
    # railed pixel. The saturation map uses the MAX across burn-in frames
    # (saturation_ref): a median stack shaves intermittently-railed peaks
    # below threshold and lets them sneak in.
    sat_map = saturation_ref if saturation_ref is not None else reference_image
    sat_pix = np.argwhere(sat_map >= 0.8 * saturation_level)  # (y, x) of railed pixels
    if len(sat_pix):
        sat_tree = cKDTree(sat_pix[:, ::-1])  # -> (x, y)
        d_sat, _ = sat_tree.query(cand_xy)
        keep = d_sat > 40.0
        cand_xy, cand_flux = cand_xy[keep], cand_flux[keep]
        if len(cand_xy) == 0:
            return ([], np.empty((0, 2))) if return_all else []

    # Reject candidates near the frame edges: alignment padding (astroalign
    # zero-fill / integer roll wrap) creates static structure in the stacked
    # reference that repeats every frame and defeats the persistence test.
    h, w = reference_image.shape
    in_bounds = ((cand_xy[:, 0] > edge_margin) & (cand_xy[:, 0] < w - edge_margin) &
                 (cand_xy[:, 1] > edge_margin) & (cand_xy[:, 1] < h - edge_margin))
    cand_xy, cand_flux = cand_xy[in_bounds], cand_flux[in_bounds]
    if len(cand_xy) == 0:
        return []

    # Reject candidates in poorly-illuminated regions (vignetted corners):
    # residual flat-field structure there repeats every frame and would pass
    # the persistence test despite not being stars.
    if valid_mask is not None:
        ix = np.clip(cand_xy[:, 0].astype(int), 0, valid_mask.shape[1] - 1)
        iy = np.clip(cand_xy[:, 1].astype(int), 0, valid_mask.shape[0] - 1)
        keep = valid_mask[iy, ix]
        cand_xy, cand_flux = cand_xy[keep], cand_flux[keep]
        if len(cand_xy) == 0:
            return []

    # Stage 2: persistence confirmation in individual frames (relaxed threshold)
    n_frames = len(frames)
    min_hits = max(2, int(np.ceil(min_frac * n_frames)))

    hits = np.zeros(len(cand_xy), dtype=int)
    for f in frames:
        xy, _flux = _detect_sources(f, fwhm_estimate, confirm_sigma, saturation_level)
        if len(xy) == 0:
            continue
        d, _ = cKDTree(xy).query(cand_xy, distance_upper_bound=tolerance)
        hits += (np.isfinite(d) & (d < tolerance)).astype(int)

    persistent = hits >= min_hits
    coords = cand_xy[persistent]
    fluxes = cand_flux[persistent]

    if len(coords) == 0:
        return ([], np.empty((0, 2))) if return_all else []

    # Crowding filter: drop only the fainter member of each close pair
    tree = cKDTree(coords)
    pairs = tree.query_pairs(min_separation)
    crowded = set()
    for i, j in pairs:
        crowded.add(i if fluxes[i] < fluxes[j] else j)

    order = np.argsort(fluxes)[::-1]  # brightest first
    selected = []
    for i in order:
        if i not in crowded:
            selected.append(tuple(coords[i]))
            if len(selected) >= max_stars:
                break

    if return_all:
        # ALL persistent star positions (pre-crowding, pre-cap) — used by
        # Engine A's known-star veto. Subtraction residuals of faint stars
        # beyond the tracking cap persist frame after frame, survive temporal
        # verification, and were a chronic source of huge-sigma junk alerts.
        return selected, coords
    return selected


def find_stars_autonomously(image, fwhm_estimate=3.0, threshold_sigma=5.0, max_stars=500, saturation_level=55000.0, min_separation=16.0):

    raw_coords, _fluxes = _detect_sources(image, fwhm_estimate, threshold_sigma, saturation_level)

    if len(raw_coords) == 0:
        return []

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

def optimal_image_subtraction(target_image, reference_image, psf_kernel=None, bg_diff=0.0, cache=None, refit_interval=10):
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
        # Seeing evolves on minute timescales, not frame timescales, so the
        # kernel solve (the expensive part) can be reused across frames.
        # `cache` (a dict owned by the caller) holds the last solution;
        # a full bidirectional refit runs every `refit_interval` frames.
        if cache is not None and cache.get('age', refit_interval) < refit_interval:
            cache['age'] += 1
            K, bg, direction = cache['K'], cache['bg'], cache['direction']
        else:
            # Calculate the dynamic atmospheric blur and background offset in BOTH directions
            # This handles the case where the reference image is blurrier than the target image,
            # preventing massive ringing artifacts from unstable deconvolution.
            K1, bg1 = fit_optimal_kernel(target_image, reference_image, kernel_size=5)
            K2, bg2 = fit_optimal_kernel(reference_image, target_image, kernel_size=5)

            # Less negative mass means a more physically stable blurring kernel (closer to 0 is better).
            if np.sum(K1[K1 < 0]) > np.sum(K2[K2 < 0]):
                K, bg, direction = K1, bg1, 'blur_ref'
            else:
                K, bg, direction = K2, bg2, 'blur_target'
            if cache is not None:
                cache.update(K=K, bg=bg, direction=direction, age=0)

        if direction == 'blur_ref':
            # Reference is sharper. Blur reference to match target.
            with scipy.fft.set_workers(FFT_WORKERS):
                convolved_ref = fftconvolve(reference_image, K, mode='same')
            difference_image = target_image - convolved_ref - bg
        else:
            # Target is sharper. Blur target to match reference.
            with scipy.fft.set_workers(FFT_WORKERS):
                convolved_target = fftconvolve(target_image, K, mode='same')
            difference_image = convolved_target - reference_image - bg

    else:
        # Artificially blur the reference image using Fast Fourier Transform convolution (Multi-Core)
        with scipy.fft.set_workers(FFT_WORKERS):
            convolved_ref = fftconvolve(reference_image, psf_kernel, mode='same')
        
        # Subtract to isolate transients, applying the background offset
        difference_image = target_image - convolved_ref - bg_diff
    
    return difference_image

def extract_sources_from_difference(difference_image, background_sigma=20.0, edge_margin=100,
                                    return_local_sigma=False):
    """
    Scans the subtracted difference image to find statistically significant clusters
    of glowing pixels that survived the subtraction process.
    
    Args:
        difference_image: 2D numpy array (the output of Engine A)
        background_sigma: The SNR threshold required to trigger an extraction.
                          (e.g. 5.0 means the object must be 5x brighter than the noise floor)
        edge_margin: Number of pixels from the edge of the image to ignore.
                     (Ignores artifacts from zero-padding during image alignment.
                     [R17] Default reduced 250 -> 100: measured alignment shifts
                     on real data are <5 px; 250 discarded ~24% of the frame.)
        return_local_sigma: also return each object's significance measured
                     against the LOCAL background RMS at its position —
                     the global RMS understates noise near bright-star
                     residuals and edges, which is how glitches were being
                     reported as 500+ sigma "detections".
                          
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

    if return_local_sigma:
        if len(objects) > 0:
            rms_map = bkg.rms()
            ix = np.clip(objects['x'].round().astype(int), 0, diff_data.shape[1] - 1)
            iy = np.clip(objects['y'].round().astype(int), 0, diff_data.shape[0] - 1)
            local_rms = np.maximum(rms_map[iy, ix], 1e-6)
            local_sigma = objects['peak'] / local_rms
        else:
            local_sigma = np.empty(0)
        return objects, bkg.globalrms, local_sigma

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
    # [P1] float32: halves memory footprint and bandwidth of every
    # downstream image operation; photometric sums stay float64 internally.
    calibrated = raw_image.astype(np.float32) - master_bias.astype(np.float32, copy=False)
    
    # Divide by flat (vignetting & dust spots)
    # Zero-guard to prevent divide-by-zero on edge pixels
    flat_safe = np.where(master_flat == 0, 1.0, master_flat)
    calibrated /= flat_safe
    
    # Heal hot/dead pixels if a mask is provided.
    # [P1] Targeted healing: median-filtering the ENTIRE 15-Mpx frame to fix
    # ~6k masked pixels cost ~2 s/frame — the single largest hidden stage.
    # Compute the 3x3 median only AT the masked positions (~ms).
    if bad_pixel_mask is not None:
        if bad_pixel_mask.shape != calibrated.shape:
            print("Warning: bad_pixel_mask shape does not match image shape. Skipping mask.")
        else:
            ys, xs = np.nonzero(bad_pixel_mask)
            if len(ys):
                h, w = calibrated.shape
                neigh = np.stack([calibrated[np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)]
                                  for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=0)
                calibrated[ys, xs] = np.median(neigh, axis=0)
            
    return calibrated

def estimate_translation(target_image, reference_image, downsample=4):
    """
    Estimates the global (dy, dx) shift between two frames via phase
    cross-correlation. Runs on a downsampled copy for speed, then refines
    to full-resolution pixels. Returns (dy, dx, confidence) where confidence
    is the correlation peak height relative to the field (higher = sharper lock).
    """
    a = target_image[::downsample, ::downsample].astype(np.float64)
    b = reference_image[::downsample, ::downsample].astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()

    with scipy.fft.set_workers(FFT_WORKERS):
        Fa = scipy.fft.rfft2(a)
        Fb = scipy.fft.rfft2(b)
        cross = Fa * np.conj(Fb)
        # Normalize to phase-only: sharpens the correlation peak
        cross /= (np.abs(cross) + 1e-12)
        corr = scipy.fft.irfft2(cross, s=a.shape)

    peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
    peak_val = corr[peak_idx]
    confidence = peak_val / (np.abs(corr).mean() + 1e-12)

    dy, dx = peak_idx
    # Wrap negative shifts
    if dy > a.shape[0] // 2:
        dy -= a.shape[0]
    if dx > a.shape[1] // 2:
        dx -= a.shape[1]

    return dy * downsample, dx * downsample, confidence


def align_image(target_image, reference_image, max_translation=150, try_affine=True, return_method=False):
    """
    Aligns target to reference. Tries astroalign (full affine, needs >=3 stars);
    on sparse fields where triangle matching fails, falls back to a global
    translation from phase cross-correlation. Only raises AlignmentError
    (slew suspected) when the fallback also fails or the measured shift
    exceeds max_translation pixels — a real slew moves the field by far more
    than tracking drift ever does.

    try_affine=False skips the astroalign attempt entirely (callers use this
    after repeated affine failures on sparse fields — a failed attempt costs
    ~0.5 s per frame for nothing).
    return_method=True returns (aligned, 'affine'|'translation') instead.
    """
    aa_err = "affine skipped"
    if try_affine:
        try:
            aligned, _ = aa.register(target_image, reference_image)
            aligned = aligned.astype(np.float32, copy=False)  # [P1] keep the float32 chain
            # Affine path: drift is baked into the transform; shift unknown here
            return (aligned, 'affine', None) if return_method else aligned
        except Exception as e:
            aa_err = e

    # Sparse-field fallback: translation-only alignment.
    try:
        dy, dx, confidence = estimate_translation(target_image, reference_image)
    except Exception:
        raise AlignmentError(f"Astroalign and phase correlation both failed: {aa_err}")

    if abs(dy) > max_translation or abs(dx) > max_translation:
        raise AlignmentError(
            f"Field moved ({dy:+.0f}, {dx:+.0f}) px — beyond tracking drift. Telescope likely slewed.")

    if dy == 0 and dx == 0:
        aligned = target_image  # drift below resolution: alignment is a no-op
    else:
        # Integer roll is exact and cheap; wrapped edges fall inside the
        # edge_margin exclusion zone used during extraction.
        aligned = np.roll(target_image, (-int(dy), -int(dx)), axis=(0, 1))
    # The (dy, dx) shift maps aligned coords back to detector coords —
    # needed by defect vetting (defects are detector-fixed).
    return (aligned, 'translation', (int(dy), int(dx))) if return_method else aligned


def generate_master_reference(burn_in_frames, return_aligned=False):
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

    if return_aligned:
        return master_reference, aligned_frames
    return master_reference

# ============================================================
# src/photometry.py
# ============================================================

class PhotometryEngine:
    
    def __init__(self, z_threshold=5.0, min_std=25.0, var_threshold_multiplier=5.0, min_alert_flux=300.0,
                 flare_consecutive=2, flare_cooldown_frames=15):
        self.z_threshold = z_threshold
        self.min_alert_flux = min_alert_flux
        # [Advisor feedback] A flare must exceed threshold on N CONSECUTIVE
        # frames of this filter before alerting: a cosmic ray in a star's
        # aperture lasts one frame; a real flare lasts minutes at ~32 s
        # cadence. Cooldown stops one event re-alerting frame after frame.
        self.flare_consecutive = flare_consecutive
        self.flare_cooldown_frames = flare_cooldown_frames
        self._flare_streak = {}    # star -> consecutive frames over threshold
        self._flare_cooldown = {}  # star -> frames left in post-alert suppression
        self.var_threshold_multiplier = var_threshold_multiplier
        self.light_curves = {} # source_id (index) -> list of fluxes
        self.times = [] # per-frame observation time (float JD, or None) — shared by all stars
        self.reference_fluxes = {} # source_id -> raw flux on first frame
        self.baselines = {} # source_id -> (baseline_mean, baseline_std)
        self.already_alerted_var = set() # Prevent infinite spam for pulsators
        self.last_inflations = {} # source_id -> variance inflation factor from the latest frame
        
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

    def update_light_curves(self, fluxes, obs_time=None):
        """
        Maintains an in-memory time series of flux values for every tracked star.
        Evaluates the rolling Z-score and rolling variance for the current frame.

        Args:
            fluxes: array-like of flux values corresponding to the fixed source IDs.
            obs_time: observation time of this frame (float JD from the FITS
                      header), or None if unavailable. Stored in self.times so
                      light curves can be plotted against real time.

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

        self.times.append(obs_time)

        # Store reference fluxes on the very first frame
        if not self.reference_fluxes:
            for i, flux in enumerate(fluxes):
                self.reference_fluxes[i] = flux
        
        # Calculate the global ensemble zero-point correction.
        # Ensemble stars must be genuinely bright: near-threshold stars have
        # background-subtracted fluxes that fluctuate around zero, and their
        # ratios can drive the median to ~0 or negative — dividing by which
        # fabricates enormous fake flux spikes (observed 30x on 20251205).
        ref_vals = [rf for rf in self.reference_fluxes.values() if rf > 0]
        bright_cut = max(100.0, np.percentile(ref_vals, 75)) if ref_vals else 100.0
        ratios = []
        for i, flux in enumerate(fluxes):
            ref_flux = self.reference_fluxes.get(i, 0.0)
            if ref_flux >= bright_cut:
                r = flux / ref_flux
                # Discard individually absurd ratios (aperture landed on noise)
                if 0.1 < r < 10.0:
                    ratios.append(r)

        # Determine the global flux correction factor (median of all valid ratios)
        # If no history exists yet (first frame), factor is 1.0
        if len(ratios) >= 5:
            correction_factor = np.median(ratios)
        else:
            correction_factor = 1.0

        # Guard against zero/NaN and implausible transparency swings: a real
        # cloud dims the field, it does not brighten it 5x or dim it 5x in
        # one 30 s cadence step.
        if not np.isfinite(correction_factor) or not (0.2 < correction_factor < 5.0):
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
        # [R6] Rolling window for all variance estimates (~30 min at the
        # measured ~30 s per-filter cadence)
        VAR_WINDOW = 60

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
                # Flux floor: stars near the sky-noise level (aperture flux of a
                # few tens of ADU) produce z excursions that are pure noise —
                # they may be tracked for completeness but must not alert.
                over = abs(z) > self.z_threshold and b_mean >= self.min_alert_flux
                self._flare_streak[i] = self._flare_streak.get(i, 0) + 1 if over else 0
                cd = self._flare_cooldown.get(i, 0)
                if cd > 0:
                    self._flare_cooldown[i] = cd - 1
                if self._flare_streak[i] >= self.flare_consecutive and cd == 0:
                    z_alerts.append(i)
                    self._flare_cooldown[i] = self.flare_cooldown_frames

                # Pre-compute relative variance for all sources with enough history
                # (used in Pass 2 below for the pulsator check).
                # [R6] Variance on a ROLLING window — the full history compared
                # against a 10-frame baseline grows monotonically with any slow
                # drift, so false positives increased with runtime.
                if len(history) >= MIN_PULSATOR_FRAMES and abs(b_mean) > 1.0:
                    relative_vars[i] = np.var(history[-VAR_WINDOW:]) / (b_mean ** 2)

            else:
                z = 0.0
                stds.append(0.0)

            z_scores.append(z)

        # --- WEATHER GATE ---
        # Patchy clouds move stars away from their clear-sky baselines
        # non-uniformly, so the global ensemble correction cannot cancel them
        # and individually-legitimate z-spikes appear across the field
        # (measured: 148 'flares' on the partly-cloudy Spica hour). Real
        # astrophysics is rare; weather is field-wide. If an abnormal
        # fraction of baselined stars deviate simultaneously, the frame is
        # weather — suppress flare alerts for this frame only.
        baselined = [z for z, sd in zip(z_scores, stds) if sd > 0]
        if len(baselined) >= 50:
            frac_deviant = np.mean(np.abs(baselined) > 3.0)
            if frac_deviant > 0.05:
                if z_alerts:
                    print(f"WEATHER GATE: {100*frac_deviant:.0f}% of stars deviate >3 sigma — "
                          f"suppressing {len(z_alerts)} flare alert(s) this frame.")
                for i in z_alerts:
                    self._flare_cooldown[i] = 0  # refund: no alert was delivered
                z_alerts = []

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
                    measured_var = np.var(history[-VAR_WINDOW:])  # [R6] rolling window
                    baseline_var = b_std ** 2
                    
                    # Inflation factor: how much has this star's variance grown 
                    # compared to its quiet 10-frame baseline?
                    inflations[i] = measured_var / baseline_var

        self.last_inflations = inflations  # exposed for alert significance [R7]

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
        self._transient_coords = []     # (x, y, t_added) from discoveries.csv, refreshed live
        self.alert_fade_seconds = 1800  # [P2] markers age off the display after 30 min
        self._csv_offset       = 0      # byte offset for incremental CSV reads
        self._csv_col_idx      = None   # cached (x_idx, y_idx) header positions
        
        # Time Travel State
        # maxlen bounds GUI memory: each entry holds a full-resolution frame
        # (float32 ~58 MB at 4788x3194). 20 frames ~ 1.2 GB and ~10 minutes
        # of scroll-back; the previous maxlen=100 of float64 frames could
        # grow past 11 GB over a long night.
        self._history_cache    = collections.deque(maxlen=20)
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
        # [P2] Banner lives in the far-left corner — nothing centered means
        # nothing can overlap the status line or the buttons.
        self.fig.text(0.006, 0.984, "VarWatch", color='white',
                      fontsize=13, fontweight='bold', va='top')
        self.fig.text(0.006, 0.962, "MIRA transient pipeline", color='#888899',
                      fontsize=7, va='top')

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
        self._circle_collection = None
        self._zscale_cache = {}

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
            "Click a star to inspect · Pause feed = hold new files · Clear = reset field (archives alerts) · Stop night = end processing",
            ha='center', va='center', fontsize=10,
            color='#888899', transform=self.ax_info.transAxes
        )

        # Live-status badge (top-right corner of info bar)
        self._status_badge = self.ax_info.text(
            0.99, 0.5, "● IDLE",
            ha='right', va='center', fontsize=9,
            color='#555566', transform=self.ax_info.transAxes
        )

        # ---- [A8]/[P2] Pause / Clear / Stop controls (top-right) ----
        ax_pause = self.fig.add_axes([0.680, 0.965, 0.098, 0.026])
        ax_clear = self.fig.add_axes([0.788, 0.965, 0.090, 0.026])
        ax_stop  = self.fig.add_axes([0.888, 0.965, 0.098, 0.026])
        self._btn_pause = Button(ax_pause, 'Pause feed', color='#22224a', hovercolor='#333366')
        self._btn_clear = Button(ax_clear, 'Clear', color='#4a4a22', hovercolor='#666633')
        self._btn_stop  = Button(ax_stop,  'Stop night', color='#4a2222', hovercolor='#663333')
        for b in (self._btn_pause, self._btn_clear, self._btn_stop):
            b.label.set_color('#ccccdd')
            b.label.set_fontsize(9)
        self._btn_pause.on_clicked(self._on_pause_clicked)
        self._btn_clear.on_clicked(self._on_clear_clicked)
        self._btn_stop.on_clicked(self._on_stop_clicked)

        # ---- [A13] Filter selector (auto = latest processed filter) ----
        # [P2] Positioned BELOW the title strip so nothing overlays the text
        ax_filt = self.fig.add_axes([0.006, 0.770, 0.050, 0.095])
        ax_filt.set_facecolor('#15152a')
        self._filter_radio = RadioButtons(ax_filt, ('auto', 'u', 'g', 'r', 'i'), active=0)
        for lbl in self._filter_radio.labels:
            lbl.set_color('#ccccdd')
            lbl.set_fontsize(8)
        self._filter_radio.on_clicked(self._on_filter_selected)

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
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)

    # ------------------------------------------------------------------
    # [A8] Pause / Stop button handlers
    # ------------------------------------------------------------------
    def _on_pause_clicked(self, _event):
        """Toggles ingestion pause on the orchestrator (files keep arriving
        in the spool; the pipeline just stops picking them up)."""
        if self.orc.pause_event.is_set():
            self.orc.pause_event.clear()
            self._btn_pause.label.set_text('Pause feed')
        else:
            self.orc.pause_event.set()
            self._btn_pause.label.set_text('Resume feed')
        self.fig.canvas.draw_idle()

    def _on_clear_clicked(self, _event):
        """[P2] Manual field reset: archives discoveries.csv, clears light
        curves and alert markers, starts a fresh burn-in. Applied by the
        watchdog between frames (thread-safe)."""
        self.orc.clear_event.set()
        self._transient_coords = []
        self._csv_offset = 0
        self._csv_col_idx = None
        self._selected_star_id = None
        self._history_cache.clear()
        self._history_offset = 0
        self._last_alert_set = set()
        self._set_status("● CLEARING", '#e8b84b')
        self.fig.canvas.draw_idle()

    def _on_stop_clicked(self, _event):
        """Permanently stops the watchdog loop. The window stays open for
        inspecting the data collected so far."""
        self.orc.stop_event.set()
        self._btn_stop.label.set_text('Stopped')  # [P2] label reflects state
        self._set_status("● STOPPED", '#ff4444')
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # [A13] Filter selector handler
    # ------------------------------------------------------------------
    def _on_filter_selected(self, label):
        """Switches which filter stream the image + light curve display."""
        self.orc.display_filter = None if label == 'auto' else label
        # Full display reset: star indices, histories and light curves are
        # per-stream and don't transfer across filters.
        self._selected_star_id = None
        self._last_frame_id = None
        self._history_cache.clear()
        self._history_offset = 0
        self._zscale_cache = {}
        info = self.orc._last_frame_info
        if info is not None:
            self._refresh_image(info['image'], self.orc, info)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # [A6] Scroll-wheel zoom (centered on the cursor); 'r' resets the view
    # ------------------------------------------------------------------
    def _on_scroll(self, event):
        if event.inaxes != self.ax_img or event.xdata is None:
            return
        factor = 1.25 if event.button == 'up' else 0.8
        cx, cy = event.xdata, event.ydata
        x0, x1 = self.ax_img.get_xlim()
        y0, y1 = self.ax_img.get_ylim()
        self.ax_img.set_xlim(cx - (cx - x0) / factor, cx + (x1 - cx) / factor)
        self.ax_img.set_ylim(cy - (cy - y0) / factor, cy + (y1 - cy) / factor)
        self.fig.canvas.draw_idle()
        
    def _on_key_press(self, event):
        if event.key == 'r':
            # [A6] Reset zoom to the full frame
            self.ax_img.autoscale()
            if self._img_handle is not None:
                self.ax_img.set_xlim(-0.5, self._img_handle.get_array().shape[1] - 0.5)
                self.ax_img.set_ylim(self._img_handle.get_array().shape[0] - 0.5, -0.5)
            self.fig.canvas.draw_idle()
            return

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
        hist_info = self._history_cache[idx]
        self._refresh_image(hist_info['image'], self.orc, hist_info)

    # ------------------------------------------------------------------
    # FuncAnimation tick — runs on the main thread every POLL_INTERVAL_MS
    # ------------------------------------------------------------------
    def _tick(self, frame_number):
        """Called by FuncAnimation. Reads the orchestrator's shared state
        and redraws only if something has actually changed."""
        orc = self.orc

        # Stopped state wins over everything
        if orc.stop_event.is_set():
            self._set_status("● STOPPED", '#ff4444')
            return

        # Nothing rendered yet — orchestrator is still in burn-in
        info = getattr(orc, '_last_frame_info', None)
        if orc.reference_image is None or len(orc.background_stars_xy) == 0 or info is None:
            self._set_status("● PAUSED", '#e8b84b') if orc.pause_event.is_set() \
                else self._set_status("● BURN-IN", '#e8b84b')
            return

        # The frame-info dict is replaced wholesale per frame, so its identity
        # marks a new frame (an id() on the raw array could be recycled).
        frame_id = id(info)

        # Always update the underlying history cache when the orchestrator produces a new frame
        if frame_id != self._last_frame_id:
            self._last_frame_id = frame_id
            self._frame_counter += 1
            self._history_cache.append(info)

            # If we are live (not paused and looking at the newest frame), auto-redraw
            if not self._is_paused and self._history_offset == 0:
                self._refresh_image(info['image'], orc, info)

        # Refresh alert coords from the live CSV every tick
        self._load_alert_coords()

        # If new alerts arrived, force an immediate circle redraw even if the image frame
        # hasn't changed — otherwise circles only appear on the NEXT camera frame
        new_alert_set = set(self._transient_coords)
        if new_alert_set != self._last_alert_set:
            self._last_alert_set = new_alert_set
            # Redraw using whatever frame is currently displayed
            display_info = info if not self._is_paused else self._history_cache[-1 + self._history_offset]
            self._refresh_image(display_info['image'], orc, display_info)

        # Update the status badge
        if orc.pause_event.is_set():
            self._set_status("● PAUSED", '#e8b84b')
        elif orc.state == "MONITORING":
            self._set_status("● MONITORING", '#44ff88')
        else:
            self._set_status(f"● {orc.state}", '#e8b84b')

    # ------------------------------------------------------------------
    # Image panel refresh
    # ------------------------------------------------------------------
    def _refresh_image(self, img, orc, info):
        """Redraws the star-field and all circle overlays.
        `info` is the orchestrator's frame-info dict (filename, frame_num,
        field_id, obs_jd, lc_len); older int-style callers get a shim."""
        if not isinstance(info, dict):
            info = {'image': img, 'filename': '?', 'frame_num': int(info),
                    'field_id': getattr(orc, 'field_id', 1), 'obs_jd': None,
                    'filter': None, 'lc_len': int(info)}

        # ZScale stretch — computed on a subsampled copy and cached per frame.
        # Full-resolution ZScale on every redraw was a major source of GUI lag.
        img_key = id(img)
        if self._zscale_cache.get('key') == img_key:
            vmin, vmax = self._zscale_cache['limits']
        else:
            try:
                vmin, vmax = ZScaleInterval().get_limits(img[::4, ::4])
            except Exception:
                vmin, vmax = np.nanmin(img), np.nanmax(img)
            self._zscale_cache = {'key': img_key, 'limits': (vmin, vmax)}

        if self._img_handle is None:
            # [A10] origin='upper' flips the vertical axis so the display maps
            # onto a sky chart the way the camera sees it. (True North-up /
            # East-left orientation needs the solved WCS — open question for
            # the advisor.)
            self._img_handle = self.ax_img.imshow(
                img, cmap='gray', origin='upper', vmin=vmin, vmax=vmax,
                interpolation='nearest', aspect='auto'
            )
        else:
            self._img_handle.set_data(img)
            self._img_handle.set_clim(vmin, vmax)

        # --- Circle overlays as ONE collection ---
        # The previous implementation created/destroyed up to max_stars
        # individual Circle patches per redraw; a single EllipseCollection
        # renders in one draw call.
        stars = np.array(orc.background_stars_xy) if len(orc.background_stars_xy) else np.empty((0, 2))
        # [P2] Only draw markers younger than alert_fade_seconds
        _now = time.time()
        live_alerts = [(ax_, ay_) for ax_, ay_, at_ in self._transient_coords
                       if _now - at_ < self.alert_fade_seconds]
        alerts = np.array(live_alerts) if live_alerts else np.empty((0, 2))

        offsets, radii, colors, lws = [], [], [], []

        if len(stars):
            if len(alerts):
                # Chebyshev (max-coordinate) distance reproduces the old
                # |dx|<8 and |dy|<8 box match
                d, _ = cKDTree(alerts).query(stars, p=np.inf, distance_upper_bound=8)
                is_transient = np.isfinite(d)
            else:
                is_transient = np.zeros(len(stars), dtype=bool)

            for i, (x, y) in enumerate(stars):
                # Base marker always shows the star's true state (alert red /
                # normal blue) — selection no longer hides it
                if is_transient[i]:
                    colors.append('#ff4444'); lws.append(2.0); radii.append(12)
                else:
                    colors.append('#4488ff'); lws.append(1.2); radii.append(10)
                offsets.append((x, y))
                if i == self._selected_star_id:
                    # [P2] additive OUTER ring for the selection
                    offsets.append((x, y))
                    colors.append('#44ff88'); lws.append(2.5); radii.append(17)

        # Engine A alerts are NEW transients (not in background_stars_xy) —
        # draw any alert not already covered by a star circle.
        if len(alerts):
            if len(stars):
                d, _ = cKDTree(stars).query(alerts, p=np.inf, distance_upper_bound=8)
                new_alerts = alerts[~np.isfinite(d)]
            else:
                new_alerts = alerts
            for (tx, ty) in new_alerts:
                offsets.append((tx, ty))
                colors.append('#ff4444'); lws.append(2.0); radii.append(12)

        if self._circle_collection is not None:
            self._circle_collection.remove()
            self._circle_collection = None
        if offsets:
            diam = np.array(radii) * 2.0
            self._circle_collection = EllipseCollection(
                widths=diam, heights=diam, angles=0, units='xy',
                offsets=np.array(offsets), transOffset=self.ax_img.transData,
                facecolors='none', edgecolors=colors, linewidths=lws, alpha=0.75
            )
            # autolim=False: adding the collection must not reset a zoomed view
            self.ax_img.add_collection(self._circle_collection, autolim=False)

        # Update frame counter title — [A11] filename + [A12] per-field frame number
        n_alerts = len(alerts)
        n_stars  = len(orc.background_stars_xy)

        status_prefix = ""
        if self._is_paused or self._history_offset < 0:
            status_prefix = f"[PAUSED - History: {self._history_offset}]  "

        filt = f" [{info['filter']}]" if info.get('filter') else ""
        self._title.set_text(
            f"{status_prefix}Field {info['field_id']} · Frame {info['frame_num']}  |  "
            f"{info['filename']}{filt}  |  Stars: {n_stars}  |  Alerts: {n_alerts}"
        )

        # Re-plot light curve for the selected star (new frame appended)
        if self._selected_star_id is not None:
            self._plot_lightcurve(self._selected_star_id, info['lc_len'] - 1)

    # ------------------------------------------------------------------
    # Light-curve panel
    # ------------------------------------------------------------------
    def _plot_lightcurve(self, star_idx, current_frame_idx):
        """Updates the light-curve panel by mutating persistent artists.
        The previous ax.clear()-and-replot per frame rebuilt the entire axes
        every tick — a large share of the GUI redraw cost."""
        lc_data = self.orc.photometry_engine.light_curves.get(star_idx, [])

        # Lazily create the reusable artists on first use
        if not hasattr(self, '_lc_line'):
            (self._lc_line,) = self.ax_lc.plot([], [], color='#4488ff', lw=1.5, alpha=0.9)
            self._lc_vline = self.ax_lc.axvline(0, color='#ff4444', lw=1.2, linestyle='--', alpha=0.7, visible=False)
            (self._lc_marker,) = self.ax_lc.plot([], [], 'o', color='#ff4444', ms=6, zorder=5)
            # TESS-style guides: unity baseline + 1/3-sigma bands from the
            # star's locked baseline scatter
            self._lc_unity = self.ax_lc.axhline(1.0, color='#888899', lw=0.8, alpha=0.6)
            self._lc_sig1 = [self.ax_lc.axhline(1.0, color='#e8b84b', lw=0.7, linestyle=':', alpha=0.55, visible=False) for _ in range(2)]
            self._lc_sig3 = [self.ax_lc.axhline(1.0, color='#ff4444', lw=0.7, linestyle=':', alpha=0.45, visible=False) for _ in range(2)]

        if not lc_data:
            self._lc_placeholder.set_text("No photometry data yet.")
            self._lc_placeholder.set_visible(True)
            self._lc_line.set_data([], [])
            self._lc_marker.set_data([], [])
            self._lc_vline.set_visible(False)
            return

        self._lc_placeholder.set_visible(False)

        # --- X axis: real observation time when the FITS headers provide it ---
        times = self.orc.photometry_engine.times
        use_time = (len(times) == len(lc_data)) and all(t is not None for t in times)
        if use_time:
            t0 = times[0]
            x = (np.asarray(times, dtype=float) - t0) * 1440.0  # JD -> minutes
            self.ax_lc.set_xlabel("Minutes since field start", color='#666677', fontsize=9)
        else:
            x = np.arange(len(lc_data))
            self.ax_lc.set_xlabel("Frame  #", color='#666677', fontsize=9)

        # Normalize to relative flux (~1.0)
        median_flux = np.median(lc_data)
        norm_lc = np.asarray(lc_data) / median_flux if median_flux > 0 else np.asarray(lc_data)

        self._lc_line.set_data(x, norm_lc)

        # Highlight current live frame
        if 0 <= current_frame_idx < len(norm_lc):
            self._lc_vline.set_xdata([x[current_frame_idx]] * 2)
            self._lc_vline.set_visible(True)
            self._lc_marker.set_data([x[current_frame_idx]], [norm_lc[current_frame_idx]])
        else:
            self._lc_vline.set_visible(False)
            self._lc_marker.set_data([], [])

        # --- Y axis: centered on 1.0 with sigma bands (TESS-style) ---
        # sigma comes from the star's locked baseline when available,
        # otherwise from the robust (MAD) scatter of the curve itself.
        baseline = self.orc.photometry_engine.baselines.get(star_idx)
        sigma_rel = None
        if baseline and median_flux > 0:
            b_mean, b_std = baseline
            if b_std > 0:
                sigma_rel = b_std / median_flux
        if sigma_rel is None:
            mad = np.median(np.abs(norm_lc - np.median(norm_lc)))
            sigma_rel = max(1.4826 * mad, 1e-4)

        for line, k in zip(self._lc_sig1 + self._lc_sig3, [+1, -1, +3, -3]):
            line.set_ydata([1.0 + k * sigma_rel] * 2)
            line.set_visible(True)

        # Limits: enough to show the 3-sigma band and any excursion in the data
        max_dev = np.max(np.abs(norm_lc - 1.0)) if len(norm_lc) else 0.0
        half = max(4.0 * sigma_rel, 1.15 * max_dev, 0.005)
        self.ax_lc.set_ylim(1.0 - half, 1.0 + half)
        if len(x) > 1 and x[-1] > x[0]:
            span = x[-1] - x[0]
            self.ax_lc.set_xlim(x[0] - 0.02 * span, x[-1] + 0.02 * span)

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
                hms = sky.ra.to_string(unit='hourangle', sep=':', precision=2, pad=True)
                dms = sky.dec.to_string(sep=':', precision=1, alwayssign=True, pad=True)
                coord_text += f"  |  RA {hms}  Dec {dms}  ({sky.ra.deg:.5f}°, {sky.dec.deg:.5f}°)"
            except Exception:
                coord_text += "  |  RA/Dec: Error"
        else:
            coord_text += "  |  has not finished rendering"

        self._info_text.set_text(coord_text)
        self._info_text.set_color('#44ff88')

        self._plot_lightcurve(min_idx, len(self.orc.photometry_engine.times) - 1)
        # [P2] Immediate visual feedback: redraw the selection ring NOW instead
        # of waiting for the next frame tick.
        info = self.orc._last_frame_info
        if info is not None:
            self._refresh_image(info['image'], self.orc, info)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _set_status(self, label, color):
        self._status_badge.set_text(label)
        self._status_badge.set_color(color)

    def _load_alert_coords(self):
        """Incremental CSV tail-read: only bytes appended since the last tick
        are parsed. Re-reading the whole file every 1.5 s scaled O(alerts)
        per tick forever. Detects rotation (file shrinks) and restarts."""
        alerts_path = os.path.join(
            getattr(self.orc.alert_logger, 'output_dir', 'pipeline_discoveries'),
            'discoveries.csv'
        )
        if not os.path.exists(alerts_path):
            self._transient_coords = []
            self._csv_offset = 0
            self._csv_col_idx = None
            return
        try:
            size = os.path.getsize(alerts_path)
            if size < self._csv_offset:
                # File rotated (slew) — start over
                self._transient_coords = []
                self._csv_offset = 0
                self._csv_col_idx = None
            if size == self._csv_offset:
                return  # nothing new

            with open(alerts_path, 'r', newline='') as f:
                f.seek(self._csv_offset)
                chunk = f.read()
                new_offset = f.tell()

            lines = chunk.splitlines(keepends=True)
            # Drop a trailing partial line (writer mid-append); re-read next tick
            if lines and not lines[-1].endswith('\n'):
                new_offset -= len(lines[-1].encode())
                lines = lines[:-1]

            rows = list(csv.reader(lines))
            if self._csv_col_idx is None:
                if not rows:
                    return
                headers = rows.pop(0)
                try:
                    self._csv_col_idx = (headers.index('X_Pixel'), headers.index('Y_Pixel'))
                except ValueError:
                    return  # Headerless or corrupt file — skip silently

            x_idx, y_idx = self._csv_col_idx
            for row in rows:
                try:
                    if len(row) > max(x_idx, y_idx):
                        # [P2] carry arrival time so markers can age out
                        self._transient_coords.append((float(row[x_idx]), float(row[y_idx]), time.time()))
                except (ValueError, IndexError):
                    pass
            self._csv_offset = new_offset
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

class FilterStream:
    """
    Per-filter pipeline state ([A13]).

    The telescope cycles filters within a sequence (measured on 20251205:
    u -> g -> r -> i every ~31 s), so consecutive FILES on disk are different
    filters of the same pointing. Every piece of state that assumes 'the
    previous frame looks like this frame' must therefore live per-filter:
    reference image (PSF and depth differ per filter), star list, photometry
    histories, temporal verification, and the subtraction kernel cache.
    """
    def __init__(self, pe_kwargs=None):
        self.state = "BURN_IN"
        self.burn_in_cache = []
        self.frames_since_full_affine = 999  # force affine on the first monitoring frame
        self.anchor_filepath = None  # file of burn-in frame 0 — the frame all others are registered to
        self.reference_image = None
        self.background_stars_xy = []
        self.photometry_engine = PhotometryEngine(**(pe_kwargs or {}))
        self.temporal_verifier = TemporalVerifier(required_consecutive=3)
        self.veto_stars_xy = np.empty((0, 2))  # ALL persistent stars (uncapped) for Engine A veto
        self.alignment_failures = 0
        self.affine_fail_streak = 0
        self.frames_since_affine_retry = 0
        self.subtraction_cache = {}


class Orchestrator:
    """
    The Master Hardware Loop.
    Because the camera cannot talk to the telescope or the pipeline, this script acts
    as an autonomous daemon. It continuously polls a spool folder for new FITS images,
    pushes them through the engines, and gracefully handles telescope slewing.
    Frames are routed to per-filter FilterStreams by their header FILTER keyword.
    """
    def __init__(self, spool_directory="camera_spool", search_pattern="*fit*", flat=None, bias=None,
                 hot_pixels_path=None, alert_output_dir=None, flats=None, subtract_every=1, pe_kwargs=None):
        self.spool_directory = spool_directory
        self.search_pattern = search_pattern

        # Memory Leak Fix: bounded deque for eviction order + set for O(1)
        # membership tests (the `in` check runs per file per 2 s poll).
        self.processed_files = deque(maxlen=2000)
        self._processed_set = set()

        # Hardware calibration masters.
        # `flats` maps filter name -> master flat ([A13] per-filter calibration);
        # legacy single `flat` becomes the fallback for every filter ('*').
        # Each flat is normalized to unity median so calibrated images keep
        # their native ADU scale ([R4]); pixels below 50% illumination are
        # excluded from star selection (static flat structure there defeats
        # the persistence test).
        self._flats = {}        # filter -> normalized flat
        self._illum_masks = {}  # filter -> bool mask
        raw_flats = dict(flats) if flats else {}
        if flat is not None and '*' not in raw_flats:
            raw_flats['*'] = flat
        for filt, fl in raw_flats.items():
            if fl is not None and np.median(fl) > 0:
                fn = (fl / np.median(fl)).astype(np.float32)  # [P1] float32 image math
                self._flats[filt] = fn
                self._illum_masks[filt] = fn > 0.5
        self.bias = (bias if bias is not None else np.zeros((1, 1))).astype(np.float32)
       
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

        # [P0.4] KD-tree of masked hot-pixel positions (detector coords).
        # When tracking drift carries a star's aperture onto a masked pixel,
        # the healing step eats part of the star's flux -> false flare alert.
        # Flare alerts from such star/frame combinations are suppressed.
        if self.bad_pixel_mask is not None and self.bad_pixel_mask.any():
            hp = np.argwhere(self.bad_pixel_mask)  # (y, x)
            self._hot_pixel_tree = cKDTree(hp[:, ::-1])  # -> (x, y)
        else:
            self._hot_pixel_tree = None
           
        # Per-filter pipeline streams ([A13]), created lazily as filters appear
        self.streams = {}           # filter name -> FilterStream
        self.display_filter = None  # GUI-selected filter; None = latest
        self._latest_filter = None
        self._null_engine = PhotometryEngine()  # safe proxy target before any stream exists

        self.current_wcs = None     # shared across filters (same pointing)
        self._wcs_solve_started = False
        self.alert_logger = AlertLogger(output_dir=alert_output_dir)

        # Shared state read by the LiveDashboard on the main thread.
        # Written only inside process_new_image() on the daemon thread.
        self._frame_infos = {}      # filter -> latest frame-info dict

        # Field & frame bookkeeping ([A11]/[A12]): frame_number resets on
        # every slew; field_id increments so the GUI can show 'Field 2 - Frame 7'
        self.field_id = 1
        self.frame_number = 0
        self.subtract_every = subtract_every
        self._pe_kwargs = dict(pe_kwargs) if pe_kwargs else {}

        # GUI controls ([A8]): pause halts ingestion of new files (poll
        # continues); stop ends the watchdog loop entirely.
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self.clear_event = threading.Event()  # [P2] GUI 'Clear' -> safe reset between frames

    # ---- Stream routing & GUI proxies -------------------------------------
    def _get_stream(self, filt):
        key = filt or 'unknown'
        if key not in self.streams:
            logging.info(f"New filter stream: '{key}'")
            self.streams[key] = FilterStream(pe_kwargs=self._pe_kwargs)
        return self.streams[key]

    def _flat_for(self, filt):
        """(normalized_flat, illum_mask) for a filter; '*' is the legacy fallback."""
        if filt in self._flats:
            return self._flats[filt], self._illum_masks.get(filt)
        if '*' in self._flats:
            return self._flats['*'], self._illum_masks.get('*')
        return None, None

    @property
    def display_stream(self):
        """The stream the GUI is looking at: explicit selection, else latest."""
        key = self.display_filter if self.display_filter in self.streams else self._latest_filter
        return self.streams.get(key)

    # Read-only proxies so the LiveDashboard keeps its simple orc.* accessors
    @property
    def state(self):
        s = self.display_stream
        return s.state if s else "BURN_IN"

    @property
    def reference_image(self):
        s = self.display_stream
        return s.reference_image if s else None

    @property
    def background_stars_xy(self):
        s = self.display_stream
        return s.background_stars_xy if s else []

    @property
    def photometry_engine(self):
        s = self.display_stream
        return s.photometry_engine if s else self._null_engine

    @property
    def _last_frame_info(self):
        key = self.display_filter if self.display_filter in self.streams else self._latest_filter
        return self._frame_infos.get(key)

    def reset_pipeline(self, reason):
        """Called when astroalign detects the telescope has moved.
        A slew moves the pointing for EVERY filter, so all streams reset."""
        logging.warning(f"[SYSTEM RESET] {reason}")
        logging.info("Flushing all filter streams and initiating new Burn-In Phase...")
        self.field_id += 1
        self.frame_number = 0  # [A12] frame numbering restarts per field
        self.streams = {}
        self._frame_infos = {}
        self._latest_filter = None
        self.current_wcs = None
        self._wcs_solve_started = False
        self.alert_logger.rotate_csv() # Archive old discoveries and start fresh

    def _async_solve_wcs(self, filepath):
        """Runs the astrometry solver in a background thread to prevent blocking
        the daemon. Retries up to 3 times on the SAME anchor file (transient
        failures — machine load, cloud-API hiccups — were previously permanent:
        one failure meant no RA/Dec for the whole field)."""
        for attempt in range(1, 4):
            logging.info(f"Starting background WCS solver (attempt {attempt}/3)...")
            wcs_result = solve_wcs_for_image(filepath)
            if wcs_result is not None:
                self.current_wcs = wcs_result
                logging.info("Background WCS lock acquired.")
                return
            logging.warning(f"Background WCS solver failed (attempt {attempt}/3).")
            if self.stop_event.is_set():
                return
            time.sleep(30)
        logging.warning("WCS solve abandoned after 3 attempts — alerts will carry X/Y pixels only.")

    @staticmethod
    def _obs_time_from_header(header):
        """Extracts the observation time as a float Julian Date, or None.
        Prefers the JD keyword; falls back to parsing DATE-OBS (assumed UT)."""
        jd = header.get('JD')
        if jd is not None:
            try:
                return float(jd)
            except (TypeError, ValueError):
                pass
        date_obs = header.get('DATE-OBS')
        if date_obs:
            try:
                dt = datetime.datetime.fromisoformat(str(date_obs))
                # Standard civil-date -> JD conversion (valid for Gregorian dates)
                a = (14 - dt.month) // 12
                y = dt.year + 4800 - a
                m = dt.month + 12 * a - 3
                jdn = dt.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
                frac = (dt.hour - 12) / 24 + dt.minute / 1440 + dt.second / 86400
                return jdn + frac
            except (ValueError, TypeError):
                pass
        return None

    def process_new_image(self, filepath, backlog=0):
        """Passes a single new image through the pipeline architecture."""
        logging.info(f"Processing: {os.path.basename(filepath)}")

        try:
            with fits.open(filepath) as hdul:
                raw_image = hdul[0].data
                header = hdul[0].header
        except Exception as e:
            logging.error(f"Failed to read FITS file. Skipping. Error: {e}")
            return

        # Observation metadata — used for stream routing, time-axis light
        # curves and alert timestamps.
        obs_jd = self._obs_time_from_header(header)
        obs_filter = str(header.get('FILTER', '')).strip() or None
        date_obs = header.get('DATE-OBS')

        # DEFINITIVE SLEW TRIGGER: a change of target name (OBJECT header /
        # filename) means the telescope has slewed — no need to wait for
        # three alignment failures. Alignment strikes remain as backup for
        # unannounced moves within the same target name.
        obj_name = str(header.get('OBJECT', '')).strip()
        if obj_name and getattr(self, '_current_object', None) not in (None, obj_name):
            self.reset_pipeline(f"New target '{obj_name}' (was '{self._current_object}') — slew from sequence.")
        if obj_name:
            self._current_object = obj_name

        self.frame_number += 1

        # Route to this filter's pipeline stream ([A13])
        stream = self._get_stream(obs_filter)
        flat, illum_mask = self._flat_for(obs_filter or 'unknown')

        # Validate calibration shapes against the data
        if flat is None or flat.shape != raw_image.shape:
            if flat is not None:
                # [R3] A real master flat that doesn't match the data is a
                # configuration error (binning/ROI mismatch) — say so loudly
                # instead of silently discarding the calibration.
                logging.error(f"CALIBRATION MISMATCH [{obs_filter}]: flat shape {flat.shape} != image {raw_image.shape}. "
                              f"Proceeding UNCALIBRATED. Check binning/ROI of the masters!")
            flat = np.ones_like(raw_image, dtype=np.float32)
            illum_mask = None
        bias = self.bias if self.bias.shape == raw_image.shape else np.zeros_like(raw_image, dtype=np.float32)

        _t0 = time.perf_counter()
        clean_image = calibrate_image(raw_image, bias, flat, bad_pixel_mask=self.bad_pixel_mask)
        _t_cal = time.perf_counter()

        # ---------------------------------------------------------
        # PHASE 1: BURN-IN (per filter stream)
        # ---------------------------------------------------------
        if stream.state == "BURN_IN":
            stream.burn_in_cache.append(clean_image)
            if len(stream.burn_in_cache) == 1:
                stream.anchor_filepath = filepath  # everything registers to this frame
            logging.info(f"Burn-In [{obs_filter}]: Frame {len(stream.burn_in_cache)}/5 collected.")

            if len(stream.burn_in_cache) == 5:
                logging.info(f"Burn-In [{obs_filter}] Complete! Generating Dynamic Reference...")
                try:
                    # File 1: Generate Master Reference
                    stream.reference_image, aligned_burn_in = generate_master_reference(stream.burn_in_cache, return_aligned=True)

                    # File 2: Autonomously map the stars.
                    # Cross-frame persistence selection: keeps only sources that
                    # repeat across the burn-in frames, rejecting the noise peaks
                    # that dominate single-frame 5-sigma detection lists.
                    # MAX across burn-in frames: preserves railed peaks that a
                    # median would suppress (for the saturation exclusion zone)
                    sat_ref = np.max(np.array(aligned_burn_in), axis=0)
                    stream.background_stars_xy, stream.veto_stars_xy = select_persistent_stars(
                        aligned_burn_in, reference_image=stream.reference_image,
                        valid_mask=illum_mask, return_all=True, saturation_ref=sat_ref)

                    # Augment the veto list with a shape-agnostic extraction of
                    # the reference: DAO's sharpness/roundness cuts exclude
                    # distorted-PSF stars (field edges, blends, saturation)
                    # from the candidate list, but their subtraction residuals
                    # still fire huge-sigma alerts. The veto must cover
                    # ANYTHING persistently present, whatever its shape.
                    try:
                        ref_data = np.ascontiguousarray(stream.reference_image, dtype=np.float64)
                        ref_bkg = sep.Background(ref_data)
                        sep.set_sub_object_limit(4096)
                        ref_src = sep.extract(ref_data - ref_bkg.back(), 3.0 * ref_bkg.globalrms)
                        sep_xy = np.column_stack([ref_src['x'], ref_src['y']]) if len(ref_src) else np.empty((0, 2))
                        stream.veto_stars_xy = np.vstack([np.asarray(stream.veto_stars_xy).reshape(-1, 2), sep_xy])
                        logging.info(f"[{obs_filter}] Veto list: {len(stream.veto_stars_xy)} persistent/reference sources.")
                    except Exception as e:
                        logging.warning(f"[{obs_filter}] sep veto augmentation failed: {e}")
                    if len(stream.background_stars_xy) < 3:
                        # Persistence needs >=2 usable frames; fall back to
                        # single-image detection on the reference stack.
                        logging.warning(f"[{obs_filter}] Persistence selection found <3 stars; falling back to single-frame detection.")
                        stream.background_stars_xy = find_stars_autonomously(stream.reference_image)
                        stream.veto_stars_xy = np.array(stream.background_stars_xy) if stream.background_stars_xy else np.empty((0, 2))
                    logging.info(f"StarFinder [{obs_filter}] locked onto {len(stream.background_stars_xy)} persistent background stars.")

                    # File 3: Attempt to generate World Coordinate System asynchronously.
                    # The WCS is shared across filters (same pointing) — solve once
                    # per field. [R2] fix: solve on the stream's ANCHOR frame
                    # (burn-in frame 0), because every subsequent frame is
                    # registered to it — solving on the 5th file offset all
                    # reported RA/Dec by the drift between frames 1 and 5.
                    if not self._wcs_solve_started:
                        self._wcs_solve_started = True
                        self._wcs_anchor_filepath = stream.anchor_filepath or filepath
                        threading.Thread(target=self._async_solve_wcs, args=(self._wcs_anchor_filepath,), daemon=True).start()

                    stream.state = "MONITORING"
                    stream.burn_in_cache = []  # release memory; reference is built
                except AlignmentError as e:
                    self.reset_pipeline(f"Telescope slewed during Burn-In: {e}")
            return # Wait for next frame

        # ---------------------------------------------------------
        # PHASE 2: CONTINUOUS MONITORING (Engines A & B)
        # ---------------------------------------------------------
        try:
            # First, align the current frame to this filter's reference.
            # Skip the doomed astroalign attempt on fields where it keeps
            # failing (sparse star fields); re-try it every 100 frames.
            skip_affine = stream.affine_fail_streak >= 3 and stream.frames_since_affine_retry < 100
            if not skip_affine:
                stream.frames_since_affine_retry = 0
            # [P1] Affine refresh interval: a successful astroalign costs
            # ~1.5-2 s/frame, but inter-frame drift is a pure translation at
            # tracking timescales. Run the full affine solve every 10th frame
            # to absorb slow rotation; use the ~0.2 s phase-correlation
            # translation in between.
            want_affine = stream.frames_since_full_affine >= 10
            aligned_image, method, align_shift = align_image(clean_image, stream.reference_image,
                                                             try_affine=(want_affine and not skip_affine),
                                                             return_method=True)
            stream.frames_since_full_affine += 1
            if method == 'affine':
                stream.affine_fail_streak = 0
                stream.frames_since_full_affine = 0
            elif want_affine and not skip_affine:
                stream.affine_fail_streak += 1
            stream.frames_since_affine_retry += 1
            stream.alignment_failures = 0  # Reset on success
        except AlignmentError as e:
            stream.alignment_failures += 1
            if stream.alignment_failures >= 3:
                # THE HARDWARE TRIGGER: The telescope moved!
                self.reset_pipeline(f"Telescope Slew Confirmed! ({e})")
                # Use this frame as frame 1 of the new burn-in for its filter
                self._get_stream(obs_filter).burn_in_cache.append(clean_image)
            else:
                logging.warning(f"Alignment failed [{obs_filter}] (Strike {stream.alignment_failures}/3). Likely a cloud or satellite trail. Dropping frame.")
            return

        _t_align = time.perf_counter()

        raw_candidates = [] # Stores dicts: x, y, engine, sig, bypass_bouncer

        # -- ENGINE B (Photometry) --
        fluxes = stream.photometry_engine.perform_aperture_photometry(aligned_image, stream.background_stars_xy)
        z_scores, stds, z_alerts, var_alerts = stream.photometry_engine.update_light_curves(fluxes, obs_time=obs_jd)

        # Frame-info packet for the LiveDashboard ([A11]/[A12]). Single dict
        # assignment — atomic under the GIL.
        self._frame_infos[obs_filter or 'unknown'] = {
            # float32 copy: display-only (engines already ran on the float64
            # original above); halves the GUI history-cache memory footprint
            'image': aligned_image.astype(np.float32),
            'filename': os.path.basename(filepath),
            'frame_num': self.frame_number,
            'field_id': self.field_id,
            'obs_jd': obs_jd,
            'filter': obs_filter,
            'lc_len': len(stream.photometry_engine.times),
        }
        self._latest_filter = obs_filter or 'unknown'

        _t_phot = time.perf_counter()

        # [P0.4] Hot-pixel overlap suppression: map each alerting star to
        # detector coordinates via this frame's alignment shift and drop the
        # alert if a masked hot pixel sits inside its aperture (r=8, +1 pad).
        if z_alerts and self._hot_pixel_tree is not None:
            if align_shift is None:
                _dy, _dx, _c = estimate_translation(clean_image, stream.reference_image)
                align_shift = (int(_dy), int(_dx))
            kept = []
            for idx in z_alerts:
                sx, sy = stream.background_stars_xy[idx]
                d, _ = self._hot_pixel_tree.query([sx + align_shift[1], sy + align_shift[0]])
                if d <= 9.0:
                    logging.info(f"[{obs_filter}] Flare alert on star {idx} suppressed: "
                                 f"masked hot pixel {d:.1f}px from star center this frame.")
                else:
                    kept.append(idx)
            z_alerts = kept

        # Flares are 1-frame events. They bypass the temporal bouncer.
        for idx in z_alerts:
            x, y = stream.background_stars_xy[idx]
            sig = f"Z={z_scores[idx]:.1f}"
            raw_candidates.append({'x': x, 'y': y, 'engine': 'Engine B (Flare)', 'sig': sig, 'bypass': True, 'star_id': idx})

        # Pulsators are slow variables. They go to the bouncer.
        for idx in var_alerts:
            x, y = stream.background_stars_xy[idx]
            # [R7] Report the variance-inflation factor that actually fired the
            # alert (was: the baseline std in ADU, an unrelated number).
            inf = stream.photometry_engine.last_inflations.get(idx, 0.0)
            sig = f"VarInf={inf:.1f}x"
            raw_candidates.append({'x': x, 'y': y, 'engine': 'Engine B (Pulsator)', 'sig': sig, 'bypass': False, 'star_id': idx})

        # -- ENGINE A (Optimal Image Subtraction) --
        # [P1] Auto-catchup: when a backlog builds, skip image subtraction
        # (Engine A) — photometry stays real-time; discovery resumes when
        # caught up. --subtract-every N additionally thins Engine A on slow
        # hardware (default 1 = every frame).
        run_engine_a = backlog <= 2 and (self.frame_number % max(1, self.subtract_every) == 0)
        if not run_engine_a and backlog > 2 and self.frame_number % 10 == 0:
            logging.info(f"Catch-up mode: skipping subtraction (backlog={backlog}).")
        if run_engine_a:
            diff_image = optimal_image_subtraction(aligned_image, stream.reference_image,
                                                   cache=stream.subtraction_cache)
            new_objects, bkg_rms, local_sigmas = extract_sources_from_difference(diff_image, return_local_sigma=True)
            # Defects are DETECTOR-fixed; detections are in ALIGNED coords. Map
            # back using the alignment shift (fields can drift tens of px over a
            # night — a fixed search box misses the defect after enough drift).
            # Affine path: measure the net translation once per frame on demand.
            if len(new_objects) and align_shift is None:
                _dy, _dx, _conf = estimate_translation(clean_image, stream.reference_image)
                align_shift = (int(_dy), int(_dx))
            # Known-star veto against ALL persistent stars (uncapped list), not
            # just the 500 tracked for photometry — residuals of faint untracked
            # stars persist across frames and were passing every other check.
            veto_tree = cKDTree(stream.veto_stars_xy) if len(stream.veto_stars_xy) else None
            for obj, sigma_local in zip(new_objects, local_sigmas):
                # DIPOLE VETO: a subtraction/alignment artifact leaves a positive
                # peak paired with a comparable negative hole a few px away. A real
                # transient adds flux only — no hole. This is the main source of
                # the absurd 500+ sigma "detections".
                oy, ox = int(round(obj['y'])), int(round(obj['x']))
                box = diff_image[max(0, oy-12):oy+13, max(0, ox-12):ox+13]
                if box.size and box.min() < -0.5 * obj['peak']:
                    continue  # dipole artifact
                if spatial_profile_vetting(obj):
                    # Saturation Check: Ensure this isn't a blooming artifact from a bright star.
                    # [R5] Run against the RAW frame — the calibrated image is
                    # bias-subtracted/flat-divided, so 55000 ADU no longer means
                    # detector saturation there. Radius widened to 8 px to absorb
                    # the small alignment shift between raw and aligned coordinates.
                    # single_pixel_vetting runs on the RAW frame: alignment
                    # interpolation smears a defect across pixels and defeats the
                    # dominance test on the aligned image. Search box 6 px absorbs
                    # the raw<->aligned coordinate offset.
                    det_x = obj['x'] + (align_shift[1] if align_shift else 0)
                    det_y = obj['y'] + (align_shift[0] if align_shift else 0)
                    if saturation_vetting(det_x, det_y, raw_image, search_radius=8) \
                            and single_pixel_vetting(det_x, det_y, clean_image, search_box=5):
                        # CROSS-MATCH VETO: Is this "new" object actually just a poorly subtracted known star?
                        # Engine A is for discovering uncataloged objects in empty space.
                        # If it's a known star, Engine B handles it.
                        # Veto radius 8 px (was 3): subtraction residuals of bright
                        # stars centroid several px off the catalog position — the
                        # epsPer residual landed 4.5 px from its star and fired a
                        # false 'new object' alert at 3 px.
                        if veto_tree is not None:
                            d, _ = veto_tree.query([obj['x'], obj['y']], p=np.inf)
                            is_known = d < 8.0
                        else:
                            is_known = False
                        if not is_known:
                            # Significance against the LOCAL noise at the object's
                            # position — the global RMS understates noise near
                            # residuals/edges and produced inflated sigmas.
                            sig = f"Sigma={sigma_local:.1f}"
                            raw_candidates.append({'x': obj['x'], 'y': obj['y'], 'engine': 'Engine A (New)', 'sig': sig, 'bypass': False})
                   
        _t_engA = time.perf_counter()
        # [P1] Per-stage timing: measures where THIS machine actually spends
        # time — send this log line when reporting performance problems.
        logging.info(f"[timing {obs_filter}] cal={1000*(_t_cal-_t0):.0f}ms "
                     f"align={1000*(_t_align-_t_cal):.0f}ms photB={1000*(_t_phot-_t_align):.0f}ms "
                     f"subA={1000*(_t_engA-_t_phot):.0f}ms total={1000*(_t_engA-_t0):.0f}ms backlog={backlog}")

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
                self.alert_logger.log_alert(m['engine'], m['x'], m['y'], aligned_image, wcs=self.current_wcs,
                                            significance=m['sig'], filter_name=obs_filter, obs_jd=obs_jd, obs_datetime=date_obs, star_id=m.get('star_id'))
            else:
                bouncer_candidates.append(m)

        # To track objects temporally, we pass their raw float X,Y coordinates.
        # Each filter has its own verifier, so 'consecutive' means consecutive
        # frames of the SAME filter — not consecutive files on disk.
        coord_ids = [(float(m['x']), float(m['y'])) for m in bouncer_candidates]

        survivors = stream.temporal_verifier.verify(coord_ids)

        for survivor_id in survivors:
            # Find the original candidate data to pass to the logger
            for m in bouncer_candidates:
                if (float(m['x']), float(m['y'])) == survivor_id:
                    self.alert_logger.log_alert(m['engine'], m['x'], m['y'], aligned_image, wcs=self.current_wcs,
                                                significance=m['sig'], filter_name=obs_filter, obs_jd=obs_jd, obs_datetime=date_obs, star_id=m.get('star_id'))
                    break # Logged

    def run_watchdog(self):
        """The infinite polling loop designed for an air-gapped machine."""
        logging.info(f"Starting Orchestrator. Watching directory: {self.spool_directory}")
        if not os.path.exists(self.spool_directory):
            os.makedirs(self.spool_directory)

        while not self.stop_event.is_set():
            # [P2] Manual clear requested from the GUI: reset between frames
            if self.clear_event.is_set():
                self.clear_event.clear()
                self.reset_pipeline("Manual clear from GUI.")
            # [A8] Pause: hold ingestion but stay alive for a quick resume
            if self.pause_event.is_set():
                time.sleep(0.5)
                continue
            # Find all fits files matching the pattern, sorted alphabetically by filename
            # [P1] Chronological order (mtime), NOT alphabetical: when a new
            # target's files land during a backlog, name-sorting interleaves
            # fields and repeatedly triggers the target-change reset.
            fits_files = sorted(glob.glob(os.path.join(self.spool_directory, self.search_pattern)),
                                key=lambda fp: (os.path.getmtime(fp), fp))
           
            # I/O RACE CONDITION PATCH: File Stability Lock
            # Snapshot sizes for ALL new files, sleep once, then re-check —
            # the old per-file 0.5 s sleep serialized a backlog of N files
            # into N * 0.5 s of pure waiting.
            new_files = [fp for fp in fits_files if fp not in self._processed_set]
            if new_files:
                sizes_1 = {fp: os.path.getsize(fp) for fp in new_files}
                time.sleep(0.5)
                for k, filepath in enumerate(new_files):
                    # React to Stop/Pause promptly even mid-backlog
                    if self.stop_event.is_set() or self.pause_event.is_set():
                        break
                    try:
                        stable = os.path.getsize(filepath) == sizes_1[filepath] and sizes_1[filepath] > 0
                    except OSError:
                        continue  # file vanished mid-check
                    if stable:
                        self.process_new_image(filepath, backlog=len(new_files) - k - 1)
                        if len(self.processed_files) == self.processed_files.maxlen:
                            self._processed_set.discard(self.processed_files[0])
                        self.processed_files.append(filepath)
                        self._processed_set.add(filepath)

            # Wait before checking the folder again
            time.sleep(2)

        logging.info("Watchdog stopped by user request.")

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
    parser = argparse.ArgumentParser(description="Launch VarWatch (MIRA transient pipeline).")
    parser.add_argument("--pattern", type=str, default="*g*fit*",
                        help="Glob pattern to enforce tracking the same star and filter.")
    parser.add_argument("--filter", type=str, default="g", choices=['g', 'i', 'r'],
                        help="Which filter to load calibration frames for (g, i, or r).")
    # --- Remote-testing overrides (all optional; omitting them preserves ---
    # --- the observatory-machine behavior exactly)                       ---
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory of FITS frames to watch. Overrides the Desktop/'14-inch Photometry' prompt.")
    parser.add_argument("--calib-dir", type=str, default=None,
                        help="Directory holding master bias/flat frames. Overrides '<Extreme SSD>/calibration frames'.")
    parser.add_argument("--hot-pixels", type=str, default=None,
                        help="Path to a hot-pixels dark frame. Overrides '<Extreme SSD>/hot_pixels.fts'.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory for discoveries.csv and alert files. Overrides Desktop/pipeline_discoveries.")
    parser.add_argument("--min-alert-flux", type=float, default=300.0,
                        help="Minimum baseline flux (ADU) for a star to be eligible for flare alerts.")
    parser.add_argument("--flare-consecutive", type=int, default=2,
                        help="Consecutive frames over threshold required before a flare alerts (cosmic-ray rejection).")
    parser.add_argument("--subtract-every", type=int, default=1,
                        help="Run Engine A (image subtraction) every Nth frame (1 = every frame).")
    args = parser.parse_args()

    print("Initializing VarWatch...")

    ssd_root = get_extreme_ssd_path()
    if not ssd_root:
        print("WARNING: 'Extreme SSD' not found. Falling back to current directory.")
        ssd_root = os.getcwd()
    else:
        print(f"SUCCESS: 'Extreme SSD' mounted at {ssd_root}")

    hot_pixels_path = args.hot_pixels if args.hot_pixels else os.path.join(ssd_root, 'hot_pixels.fts')

    # --- LOAD CALIBRATION FRAMES ---
    calib_dir = args.calib_dir if args.calib_dir else os.path.join(ssd_root, 'calibration frames')
    flat_data, bias_data = None, None

    # Try .fits extension first, fall back to .fts
    def _find_calib_file(calib_dir, basename):
        for ext in ['.fits', '.fts', '.fit']:
            p = os.path.join(calib_dir, basename + ext)
            if os.path.exists(p):
                return p
        return None

    bias_path = _find_calib_file(calib_dir, 'median_bias')

    flats_data = {}
    try:
        if bias_path:
            with fits.open(bias_path) as hdul:
                bias_data = hdul[0].data.astype(float)
            print(f"Loaded master bias: {bias_path}")
        else:
            print(f"Warning: Master bias not found in '{calib_dir}' (tried .fits/.fts/.fit)")

        # [A13] Load a master flat for every filter that has one — the
        # pipeline routes each frame to its filter's calibration.
        for filt in ['u', 'g', 'r', 'i']:
            flat_path = _find_calib_file(calib_dir, f'mean_flat_{filt}')
            if flat_path:
                with fits.open(flat_path) as hdul:
                    flats_data[filt] = hdul[0].data.astype(float)
                print(f"Loaded master flat ({filt}): {flat_path}")
        if args.filter not in flats_data:
            print(f"Warning: Master flat for '{args.filter}' not found in '{calib_dir}' (tried .fits/.fts/.fit)")
        flat_data = flats_data.get(args.filter)
    except Exception as e:
        print(f"Error loading calibration frames: {e}")

    if args.data_dir:
        # Remote-testing mode: watch the directory given on the command line.
        target_dir = args.data_dir
    else:
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
        flats=flats_data,
        bias=bias_data,
        alert_output_dir=args.out_dir,
        subtract_every=args.subtract_every,
        pe_kwargs={'min_alert_flux': args.min_alert_flux, 'flare_consecutive': args.flare_consecutive}
    )
    
    orchestrator.start_live_dashboard()

if __name__ == "__main__":
    main()
