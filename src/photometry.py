import numpy as np
import sep
from photutils.aperture import CircularAperture, aperture_photometry

class PhotometryEngine:
    
    def __init__(self, z_threshold=17.0, min_std=25.0, var_threshold_multiplier=5.0):
        self.z_threshold = z_threshold
        self.var_threshold_multiplier = var_threshold_multiplier
        self.light_curves = {} # source_id (index) -> list of fluxes
        self.reference_fluxes = {} # source_id -> raw flux on first frame
        self.baselines = {} # source_id -> (baseline_mean, baseline_std)
        
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
                
            # Only flag stars that have inflated significantly MORE than the atmosphere caused
            threshold = median_inf + self.var_threshold_multiplier * robust_sigma
            
            for i, inf in inflations.items():
                if inf > threshold:
                    var_alerts.append(i)

        return np.array(z_scores), np.array(stds), z_alerts, var_alerts
