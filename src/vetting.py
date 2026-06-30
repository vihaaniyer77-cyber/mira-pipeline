import numpy as np

def saturation_vetting(x, y, raw_image, saturation_level=55000.0, search_radius=2):
    """
    The Bouncer (Saturation).
    Engine A finds transients by subtracting images. If a bright star in the raw 
    image hits the physical CCD limit (e.g., 65,535), it bleeds into adjacent pixels. 
    Subtraction math fails catastrophically on these bleeding columns, creating 
    massive false positives that look like new stars.
    
    This checks if the proposed transient is on or near a saturated pixel in the RAW image.
    
    Args:
        x, y: Pixel coordinates of the proposed transient.
        raw_image: The aligned 2D camera frame BEFORE subtraction.
        saturation_level: The ADU threshold considered dangerous.
        search_radius: How many pixels around the center to check for saturation bleeding.
        
    Returns:
        Boolean: True if safe (not saturated). False if it's a bleeding artifact.
    """
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

def spatial_profile_vetting(extracted_object, min_fwhm=1.5, max_fwhm=8.0, max_ellipticity=0.4, min_pixels=4):
    """
    The Bouncer (Spatial). 
    Analyzes the geometric shape of an alert from Engine A. 
    
    Because we operate entirely in a pixel coordinate space without WCS, we rely 
    heavily on morphology to distinguish true stars from sensor artifacts.
    
    Args:
        extracted_object: dict-like row output from the sep extraction library.
    
    Returns:
        Boolean: True if the object's geometry resembles a real stellar transient.
                 False if it is a cosmic ray, hot pixel, or satellite streak.
    """
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
    """
    The Bouncer (Temporal).
    Enforces a strict 'persistence' rule. True stellar transients do not move, 
    but slow satellites or lingering sensor artifacts might survive spatial vetting.
    
    This class tracks the (X, Y) pixel index of transients and requires them to 
    appear in the exact same spot for N consecutive frames before authorizing an alert.
    """
    def __init__(self, required_consecutive=3):
        self.required = required_consecutive
        self.history = {} # obj_id (e.g. X,Y coordinate tuple) -> consecutive count
        
    def verify(self, current_detections):
        """
        Updates the temporal history of all currently detected objects.
        
        Args:
            current_detections: list of object IDs (or coordinate tuples) in the current frame.
            
        Returns:
            valid_targets: A list of objects that have met the consecutive frame requirement.
        """
        valid_targets = []
        
        # Increment the count for objects seen in this frame
        for obj_id in current_detections:
            self.history[obj_id] = self.history.get(obj_id, 0) + 1
            if self.history[obj_id] >= self.required:
                valid_targets.append(obj_id)
                
        # Instantly reset the count for any object that disappeared
        # MEMORY LEAK PATCH: Delete the key entirely so the dictionary doesn't bloat infinitely.
        for obj_id in list(self.history.keys()):
            if obj_id not in current_detections:
                del self.history[obj_id]
                
        return valid_targets
