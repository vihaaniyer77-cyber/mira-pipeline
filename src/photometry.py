import numpy as np
from photutils.aperture import CircularAperture, aperture_photometry

class PhotometryEngine:
    """
    Engine B: The Forced Photometry Engine.
    
    This engine acts as a massive parallel tracking system. It receives a dynamic list 
    of thousands of (X, Y) pixel coordinates from the StarFinder module. It places a 
    circular aperture over every star and maintains a rolling history of their brightness.
    
    It actively hunts for two specific types of anomalies:
    1. Sudden Spikes (Flares) via a Rolling Z-Score.
    2. Slow Oscillations (Pulsators) via Rolling Variance.
    """
    def __init__(self, window_size=10, z_threshold=4.0, min_std=15.0, var_threshold_multiplier=3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.min_std = min_std
        self.var_threshold_multiplier = var_threshold_multiplier
        self.light_curves = {} # source_id (index) -> list of fluxes
        
    def perform_aperture_photometry(self, image, positions, aperture_radius=5.0):
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
            
        apertures = CircularAperture(positions, r=aperture_radius)
        phot_table = aperture_photometry(image, apertures)
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
            var_alerts: list of source_ids that triggered a Pulsator alert (Variance > threshold).
        """
        z_scores = []
        stds = []
        z_alerts = []
        var_alerts = []
        
        for i, flux in enumerate(fluxes):
            if i not in self.light_curves:
                self.light_curves[i] = []
            
            # Compute rolling statistics on the historical window before appending the current flux
            history = self.light_curves[i][-self.window_size:]
            
            if len(history) >= 2:
                mean_flux = np.mean(history)
                raw_std_flux = np.std(history)
                stds.append(raw_std_flux)
                
                # TRIGGER 1: The Pulsator Catch (Slow Variables)
                # If the raw rolling standard deviation exceeds the atmospheric noise floor multiplier
                if raw_std_flux > self.min_std * self.var_threshold_multiplier:
                    var_alerts.append(i)
                
                # Apply the noise floor to prevent dividing by an artificially small sample std
                std_flux = max(raw_std_flux, self.min_std)
                
                # Guard against exact zero std with a tiny epsilon so alerts still fire
                # when flux suddenly jumps from a perfectly constant baseline (e.g. simulations)
                z = (flux - mean_flux) / (std_flux if std_flux > 0 else 1e-10)
            else:
                # Not enough history to calculate statistics
                z = 0.0
                stds.append(0.0)
                
            z_scores.append(z)
            self.light_curves[i].append(flux)
            
            # MEMORY LEAK PATCH: Truncate history to prevent infinite RAM usage.
            # We only ever need the last `window_size` frames for math. Keeping 2x is safe.
            if len(self.light_curves[i]) > self.window_size * 2:
                self.light_curves[i] = self.light_curves[i][-self.window_size*2:]
            
            # TRIGGER 2: The Flare Catch (Sudden Spikes)
            if abs(z) > self.z_threshold:
                z_alerts.append(i)
            
        return np.array(z_scores), np.array(stds), z_alerts, var_alerts
