import os
import time
import glob
import logging
import threading
from collections import deque
from astropy.io import fits
import numpy as np

# Import our custom modules
from calibration import calibrate_image, align_image, generate_master_reference, AlignmentError
from starfinder import find_stars_autonomously
from photometry import PhotometryEngine
from subtraction import optimal_image_subtraction, extract_sources_from_difference
from vetting import spatial_profile_vetting, saturation_vetting, TemporalVerifier
from alert_logger import AlertLogger
from astrometry_solver import solve_wcs_for_image

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
    def __init__(self, spool_directory="camera_spool", flat=None, bias=None, hot_pixels_path=r"S:\Jean\Interns\Vihaan\hot_pixels.fts"):
        self.spool_directory = spool_directory
       
        # Memory Leak Fix: Use a deque with a maxlen instead of an infinitely growing set
        self.processed_files = deque(maxlen=2000)
       
        # Hardware calibration masters (Mocked as 0/1 for fallback)
        self.flat = flat if flat is not None else np.ones((1, 1))
        self.bias = bias if bias is not None else np.zeros((1, 1))
       
        # Load Bad Pixel Mask dynamically
        if os.path.exists(hot_pixels_path):
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
       
        # Initialize the Engines
        self.photometry_engine = PhotometryEngine()
        self.temporal_verifier = TemporalVerifier(required_consecutive=3)
        self.alert_logger = AlertLogger()

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
        except AlignmentError as e:
            # THE HARDWARE TRIGGER: The telescope moved!
            self.reset_pipeline(f"Telescope Slew Detected! {e}")
            self.burn_in_cache.append(clean_image) # Use this frame as frame 1 of new burn-in
            return

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
            # Find all fits files, sorted by creation time
            fits_files = sorted(glob.glob(os.path.join(self.spool_directory, "*.fits")), key=os.path.getctime)
           
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

if __name__ == "__main__":
    orchestrator = Orchestrator()
    # orchestrator.run_watchdog()
