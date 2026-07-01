import numpy as np
import astroalign as aa

class AlignmentError(Exception):
    """Custom exception raised when telescope slewing or poor seeing prevents image alignment."""
    pass

from scipy.ndimage import median_filter

def calibrate_image(raw_image, master_bias, master_dark, master_flat, exposure_time=1.0, dark_exposure=1.0, bad_pixel_mask=None):
    """
    Applies standard hardware reduction equations via vectorized matrix arithmetic.
    Removes thermal noise and optical vignetting from the raw camera sensor output.
    Standard equation in (Howell's Handbook of CCD Astronomy)
    
    Args:
        raw_image: 2D numpy array directly from the camera
        master_bias: 2D numpy array for readout noise
        master_dark: 2D numpy array for thermal noise
        master_flat: 2D numpy array for optical vignetting/dust
        bad_pixel_mask: boolean array indicating defective pixels to heal
    """
    # Subtract bias (readout noise)
    calibrated = raw_image.astype(float) - master_bias
    
    # Subtract scaled dark (thermal noise), this will be analyzed further, with specifics on whether this is actually needed
    dark_scaled = master_dark * (exposure_time / dark_exposure)
    calibrated -= dark_scaled
    
    # Divide by flat (vignetting and dust)
    # Avoid division by zero by replacing 0s with 1s in the denominator
    flat_safe = np.where(master_flat == 0, 1.0, master_flat)
    calibrated /= flat_safe
    
    # Heal defective pixels using neighborhood interpolation
    if bad_pixel_mask is not None:
        local_median = median_filter(calibrated, size=3)
        calibrated[bad_pixel_mask] = local_median[bad_pixel_mask]
        
    return calibrated

def align_image(target_image, reference_image):
    """
    Computes a rigid transformation (translation, rotation) of target_image 
    relative to reference_image by matching stellar asterisms.
    
    Raises AlignmentError if the telescope has slewed to a new target or 
    clouds have completely obscured the star field.
    """
    try:
        # astroalign registers target to reference using asterism matching
        # Limited to top 50 brightest stars to prevent infinite loops on 15-Megapixel images
        aligned_image, footprint = aa.register(target_image, reference_image, max_control_points=50)
        return aligned_image
    except aa.MaxIterError:
        # This is thrown by astroalign when it cannot find matching stars.
        # This is our hardware signal that the telescope has likely slewed!
        raise AlignmentError("Astroalign failed. The telescope likely slewed to a new target.")
    except Exception as e:
        raise AlignmentError(f"Unexpected alignment failure: {str(e)}")

def generate_master_reference(burn_in_frames):
    """
    The 'Burn-In' Phase: Dynamically generates a reference template by aligning 
    and median-stacking a sequence of frames captured at the start of observation.
    
    Args:
        burn_in_frames: list of 2D numpy arrays. The first frame is the anchor.
    """
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
            # If a single burn-in frame fails to align (e.g. a cloud passed over), we can safely skip it and build the reference from the remaining frames.
            continue
            
    if len(aligned_frames) < 3:
        # We generally want at least 3 frames for a proper median stack to reject outliers
       print("Building reference frame with fewere than 3 images. . . WARNING")
        
    # Stack into a 3D array (frames, y, x)
    stack = np.array(aligned_frames)
    
    # Median combine along the frame axis to erase transient artifacts
    master_ref = np.median(stack, axis=0)
    
    return master_ref
