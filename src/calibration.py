import numpy as np
from scipy.ndimage import median_filter
import astroalign as aa

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
