import os
import time
import glob
from astropy.io import fits
import numpy as np

# Import our 6 custom modules
from calibration import calibrate_image, align_image, generate_master_reference, AlignmentError
from starfinder import find_stars_autonomously
from photometry import PhotometryEngine
from subtraction import optimal_image_subtraction, extract_sources_from_difference
from vetting import spatial_profile_vetting, saturation_vetting, TemporalVerifier
from alert_logger import AlertLogger

class Orchestrator:
    """
    The Master Hardware Loop.
    Because the camera cannot talk to the telescope or the pipeline, this script acts 
    as an autonomous daemon. It continuously polls a spool folder for new FITS images,
    pushes them through the 6 modules, and gracefully handles telescope slewing.
    """
    def __init__(self, spool_directory="camera_spool", dark=None, flat=None, bias=None):
        self.spool_directory = spool_directory
        self.processed_files = set()
        
        # Hardware calibration masters (Mocked as 0/1 for fallback)
        self.dark = dark if dark is not None else np.zeros((1, 1))
        self.flat = flat if flat is not None else np.ones((1, 1))
        self.bias = bias if bias is not None else np.zeros((1, 1))
        
        # Pipeline State
        self.state = "BURN_IN"
        self.burn_in_cache = []
        self.reference_image = None
        self.background_stars_xy = []
        
        # Initialize the Engines
        self.photometry_engine = PhotometryEngine()
        self.temporal_verifier = TemporalVerifier(required_consecutive=3)
        self.alert_logger = AlertLogger()

    def reset_pipeline(self, reason):
        """Called when astroalign detects the telescope has moved."""
        print(f"\n[SYSTEM RESET] {reason}")
        print("Flushing cache and initiating new Burn-In Phase...")
        self.state = "BURN_IN"
        self.burn_in_cache = []
        self.reference_image = None
        self.background_stars_xy = []
        self.photometry_engine = PhotometryEngine() # Reset flux history
        self.temporal_verifier = TemporalVerifier() # Reset temporal history

    def process_new_image(self, filepath):
        """Passes a single new image through the pipeline architecture."""
        print(f"\nProcessing: {os.path.basename(filepath)}")
        
        # 1. Load and Calibrate
        try:
            with fits.open(filepath) as hdul:
                raw_data = hdul[0].data.astype(float)
        except Exception:
            print("Failed to read FITS file. Skipping.")
            return

        # Resize mock calibration frames if necessary to match data
        if self.dark.shape != raw_data.shape:
            self.dark = np.zeros_like(raw_data)
            self.flat = np.ones_like(raw_data)
            self.bias = np.zeros_like(raw_data)
            
        clean_image = calibrate_image(raw_data, self.bias, self.dark, self.flat)

        # ---------------------------------------------------------
        # PHASE 1: BURN-IN
        # ---------------------------------------------------------
        if self.state == "BURN_IN":
            self.burn_in_cache.append(clean_image)
            print(f"Burn-In Phase: Frame {len(self.burn_in_cache)}/5 collected.")
            
            if len(self.burn_in_cache) == 5:
                print("Burn-In Complete! Generating Dynamic Reference...")
                try:
                    # File 1: Generate Master Reference (will fail if telescope slewed mid-burn)
                    self.reference_image = generate_master_reference(self.burn_in_cache)
                    
                    # File 2: Autonomously map the stars
                    self.background_stars_xy = find_stars_autonomously(self.reference_image)
                    print(f"StarFinder locked onto {len(self.background_stars_xy)} background stars.")
                    
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

        current_transient_candidates = [] # Stores (X, Y, Engine Name)

        # -- ENGINE B (Photometry) --
        fluxes = self.photometry_engine.perform_aperture_photometry(aligned_image, self.background_stars_xy)
        _, _, z_alerts, var_alerts = self.photometry_engine.update_light_curves(fluxes)
        
        # Flares are 1-frame events. They bypass the temporal bouncer.
        for idx in z_alerts:
            x, y = self.background_stars_xy[idx]
            self.alert_logger.log_alert("Engine B (Flare)", x, y, aligned_image)
            
        # Pulsators are slow variables. They go to the bouncer.
        for idx in var_alerts:
            x, y = self.background_stars_xy[idx]
            current_transient_candidates.append((x, y, "Engine B (Pulsator)"))

        # -- ENGINE A (Optimal Image Subtraction) --
        diff_image = optimal_image_subtraction(aligned_image, self.reference_image)
        new_objects = extract_sources_from_difference(diff_image)
        
        for obj in new_objects:
            if spatial_profile_vetting(obj):
                # Saturation Check: Ensure this isn't a blooming artifact from a bright star
                if saturation_vetting(obj['x'], obj['y'], aligned_image):
                    current_transient_candidates.append((obj['x'], obj['y'], "Engine A (New Transient)"))

        # ---------------------------------------------------------
        # PHASE 3: TEMPORAL VETTING & LOGGING
        # ---------------------------------------------------------
        # To track objects temporally, we use their rounded integer X,Y coordinates as their "ID"
        coord_ids = [(int(round(x)), int(round(y))) for x, y, _ in current_transient_candidates]
        
        survivors = self.temporal_verifier.verify(coord_ids)
        
        for survivor_id in survivors:
            # Find the original candidate data to pass to the logger
            for x, y, engine in current_transient_candidates:
                if (int(round(x)), int(round(y))) == survivor_id:
                    self.alert_logger.log_alert(engine, x, y, aligned_image)
                    break # Logged

    def run_watchdog(self):
        """The infinite polling loop designed for an air-gapped machine."""
        print(f"Starting Orchestrator. Watching directory: {self.spool_directory}")
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
                        self.processed_files.add(filepath)
                    
            # Wait before checking the folder again
            time.sleep(2)

if __name__ == "__main__":
    # If run directly, start the watchdog
    orchestrator = Orchestrator()
    # orchestrator.run_watchdog() # Commented out so it doesn't hang the IDE during testing
