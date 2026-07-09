import numpy as np

def saturation_vetting(x, y, raw_image, saturation_level=60000.0, search_radius=2):
    
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
    
    def __init__(self, required_consecutive=3, tolerance=2.0):
        self.required = required_consecutive
        self.tolerance = tolerance
        self.history = {} # obj_id -> {'count': int, 'last_pos': (x, y)}
        self.next_id = 0
        
    def verify(self, current_detections_xy):
       
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
                # Update existing track
                self.history[best_match_id]['count'] += 1
                self.history[best_match_id]['last_pos'] = (cx, cy)
                matched_ids.add(best_match_id)
                
                if self.history[best_match_id]['count'] >= self.required:
                    valid_targets.append((cx, cy))
            else:
                # Create new track
                self.history[self.next_id] = {'count': 1, 'last_pos': (cx, cy)}
                matched_ids.add(self.next_id)
                self.next_id += 1
                
        # Instantly reset count for objects that disappeared
        # MEMORY LEAK PATCH: Delete the key entirely so the dictionary doesn't bloat infinitely.
        for obj_id in list(self.history.keys()):
            if obj_id not in matched_ids:
                del self.history[obj_id]
                
        return valid_targets
