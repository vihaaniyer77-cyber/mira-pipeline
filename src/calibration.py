import numpy as np
import astroalign as aa

class AlignmentError(Exception):
  
    pass

from scipy.ndimage import median_filter

def calibrate_image(raw_image, master_bias, master_flat, bad_pixel_mask=None):
 
    # Subtract bias (readout noise)
    calibrated = raw_image.astype(float) - master_bias
    

    
    # Divide by flat (vignetting and dust)
    # Avoid division by zero by replacing 0s with 1s in the denominator
    flat_safe = np.where(master_flat == 0, 1.0, master_flat)
    calibrated /= flat_safe
    
    # Heal defective pixels using neighborhood interpolation
    if bad_pixel_mask is not None:
        if bad_pixel_mask.shape != calibrated.shape:
            print("Warning: bad_pixel_mask shape does not match image shape. Skipping mask.")
        else:
            local_median = median_filter(calibrated, size=3)
            calibrated[bad_pixel_mask] = local_median[bad_pixel_mask]
        
    return calibrated

def align_image(target_image, reference_image):
   
    try:
        # https://astroalign.quatrope.org/en/latest/tutorial.html -- this is the tutorial I used 
        aligned_image, footprint = aa.register(target_image, reference_image, max_control_points=50)
        return aligned_image
    except aa.MaxIterError:
        # This is thrown by astroalign when it cannot find matching stars.
        # This is our hardware signal that the telescope slewed
        raise AlignmentError("Astroalign failed. The telescope likely slewed to a new target.")
    except Exception as e:
        raise AlignmentError(f"Unexpected alignment failure: {str(e)}")

def generate_master_reference(burn_in_frames):
    
    if not burn_in_frames:
        raise ValueError("Must provide at least one frame for burn-in.")
        
    anchor = burn_in_frames[0]
    aligned_frames = [anchor]
    
    # Align all subsequent frames to the first frame
    for i, frame in enumerate(burn_in_frames[1:]):
        try:
            aligned = align_image(frame, anchor)
            aligned_frames.append(aligned)
        except AlignmentError:
            # It's fine if one doiesn't work, we can just use the others
            continue
            
    if len(aligned_frames) < 3:
        # We generally want at least 3 frames for a proper median stack to reject outliers
       print("Building reference frame with fewer than 3 images. . . WARNING")
        
    
    stack = np.array(aligned_frames)
    
    
    master_ref = np.median(stack, axis=0)
    
    return master_ref
