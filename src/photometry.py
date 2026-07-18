import numpy as np
import sep
from photutils.aperture import CircularAperture, aperture_photometry

class PhotometryEngine:
    
    def __init__(self, z_threshold=17.0, min_std=25.0, var_threshold_multiplier=3.0):
        self.z_threshold = z_threshold
        self.min_std = min_std
        self.var_threshold_multiplier = var_threshold_multiplier
        self.light_curves = {} # source_id (index) -> list of fluxes
        
    def perform_aperture_photometry(self, image, positions, aperture_radius=3.0):
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
        
        # Calculate the global ensemble zero-point correction
        ratios = []
        for i, flux in enumerate(fluxes):
            if i in self.light_curves and len(self.light_curves[i]) >= 2:
                history = self.light_curves[i]
                mean_flux = np.mean(history)
                # Only use bright, stable stars for the ensemble calculation
                if mean_flux > 100.0:
                    ratios.append(flux / mean_flux)
        
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
        
        for i, flux in enumerate(corrected_fluxes):
            if i not in self.light_curves:
                self.light_curves[i] = []
            
            # Compute statistics on the entire historical baseline before appending the current flux
            history = self.light_curves[i]
            
            if len(history) >= 2:
                mean_flux = np.mean(history)
                raw_std_flux = np.std(history)
                stds.append(raw_std_flux)
                
                # We model the expected noise floor using Poisson statistics (shot noise ~ sqrt(flux))
                expected_noise = max(np.sqrt(abs(mean_flux)), self.min_std, 0.02 * abs(mean_flux))
                
                # Apply the dynamic noise floor to prevent dividing by an artificially small sample std
                std_flux = max(raw_std_flux, expected_noise)
                
                # Guard against exact zero std with a tiny epsilon
                z = (flux - mean_flux) / (std_flux if std_flux > 0 else 1e-10)
                
                # Enforce the 10-frame baseline wait before alerting!
                if len(history) >= 10:
                    # TRIGGER 1: The Variable Catch (General Variance)
                    if raw_std_flux > expected_noise * self.var_threshold_multiplier:
                        var_alerts.append(i)
                    
                    # TRIGGER 2: The Flare Catch (Sudden Spikes)
                    if abs(z) > self.z_threshold:
                        z_alerts.append(i)
            else:
                # Not enough history to calculate statistics
                z = 0.0
                stds.append(0.0)
                
            z_scores.append(z)
            self.light_curves[i].append(flux)
            
        return np.array(z_scores), np.array(stds), z_alerts, var_alerts

