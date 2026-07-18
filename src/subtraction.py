import numpy as np
import sep
from scipy.signal import fftconvolve
import scipy.fft

def fit_optimal_kernel(target, reference, kernel_size=5):
    """
    Solves the Alard-Lupton optimal kernel matching equation.
    Because atmospheric blurring ('seeing') changes constantly, we cannot simply
    subtract two images. This function calculates a spatial convolution kernel (K) 
    that mathematically matches the point spread function (PSF) of the reference 
    image to the target image.
    
    It minimizes the least-squares difference: (target - reference ⊗ K)^2
    
    Args:
        target: 2D numpy array (the current camera frame)
        reference: 2D numpy array (the dynamic burn-in reference)
        kernel_size: Integer size of the matching kernel matrix (default 5x5)
        
    Returns:
        K: The 2D convolution matrix that models the atmospheric difference.
    """
    half_k = kernel_size // 2
    
    y_min, x_min = half_k, half_k
    y_max, x_max = reference.shape[0] - half_k, reference.shape[1] - half_k
    
    stride = 10
    I_flat = target[y_min:y_max:stride, x_min:x_max:stride].flatten()
    
    num_pixels = I_flat.shape[0]
    M = np.zeros((num_pixels, kernel_size**2))
    
    col = 0
    for i in range(-half_k, half_k + 1):
        for j in range(-half_k, half_k + 1):
            # Use -i, -j to correctly match the mathematical definition of convolution
            # (which fftconvolve uses) rather than cross-correlation.
            patch = reference[y_min-i : y_max-i : stride, x_min-j : x_max-j : stride]
            M[:, col] = patch.flatten()
            col += 1
            
    # Add a column of ones to M to solve for the differential background
    M_bg = np.hstack([M, np.ones((M.shape[0], 1))])
    
    # Ridge penalty injected into the diagonal to prevent matrix singularity
    # (Ensures stability even if parts of the image are perfectly black)
    ridge = 1e-4 * np.eye(kernel_size**2)
    
    # Expand ridge penalty to include background (background gets 0 penalty)
    ridge_bg = np.zeros((kernel_size**2 + 1, kernel_size**2 + 1))
    ridge_bg[:kernel_size**2, :kernel_size**2] = ridge
    
    # INITIAL FIT
    sol = np.linalg.solve(M_bg.T @ M_bg + ridge_bg, M_bg.T @ I_flat)
    
    # --- SIGMA-CLIPPING: Exclude transient pixels from the kernel solve ---
    # We must clip based on the RESIDUALS of the initial fit, NOT the raw image!
    # Clipping the raw image throws away all the stars, which the solver needs to match PSFs.
    residuals = I_flat - (M_bg @ sol)
    mad = np.median(np.abs(residuals - np.median(residuals)))
    robust_std = 1.4826 * mad
    
    if robust_std > 0:
        good_mask = np.abs(residuals) < 5.0 * robust_std
        if good_mask.sum() >= kernel_size ** 2 + 1:
            I_flat = I_flat[good_mask]
            M_bg = M_bg[good_mask, :]
            # RE-FIT with outliers excluded
            sol = np.linalg.solve(M_bg.T @ M_bg + ridge_bg, M_bg.T @ I_flat)

    
    # Extract kernel and background offset
    k_flat = sol[:-1]
    bg_diff = sol[-1]
    
    K = k_flat.reshape((kernel_size, kernel_size))
    
    # NOTE: The forced normalization K /= K.sum() has been removed.
    # The kernel must be allowed to sum to <1 or >1 to properly scale
    # the reference image if the target image has different atmospheric transmission (clouds).
    
    return K, bg_diff

def optimal_image_subtraction(target_image, reference_image, psf_kernel=None, bg_diff=0.0):
    """
    Engine A: The Discovery Engine.
    
    This engine hunts for completely uncataloged objects (like a new supernova)
    that appear in empty space. It dynamically blurs the pristine reference image 
    to match the atmospheric distortion of the current frame, then subtracts them.
    
    Math: Difference = Target - (Reference ⊗ K) - Bkg
    
    Returns:
        difference_image: A 2D array where static stars have been mathematically 
                          erased, leaving only pure noise and new transients.
    """
    if psf_kernel is None:
        # Calculate the dynamic atmospheric blur and background offset in BOTH directions
        # This handles the case where the reference image is blurrier than the target image,
        # preventing massive ringing artifacts from unstable deconvolution.
        K1, bg1 = fit_optimal_kernel(target_image, reference_image, kernel_size=5)
        K2, bg2 = fit_optimal_kernel(reference_image, target_image, kernel_size=5)
        
        # Less negative mass means a more physically stable blurring kernel (closer to 0 is better).
        if np.sum(K1[K1 < 0]) > np.sum(K2[K2 < 0]):
            # Reference is sharper. Blur reference to match target.
            with scipy.fft.set_workers(-1):
                convolved_ref = fftconvolve(reference_image, K1, mode='same')
            difference_image = target_image - convolved_ref - bg1
        else:
            # Target is sharper. Blur target to match reference.
            with scipy.fft.set_workers(-1):
                convolved_target = fftconvolve(target_image, K2, mode='same')
            difference_image = convolved_target - reference_image - bg2
            
    else:
        # Artificially blur the reference image using Fast Fourier Transform convolution (Multi-Core)
        with scipy.fft.set_workers(-1):
            convolved_ref = fftconvolve(reference_image, psf_kernel, mode='same')
        
        # Subtract to isolate transients, applying the background offset
        difference_image = target_image - convolved_ref - bg_diff
    
    return difference_image

def extract_sources_from_difference(difference_image, background_sigma=20.0):
    """
    Scans the subtracted difference image to find statistically significant clusters
    of glowing pixels that survived the subtraction process.
    
    Args:
        difference_image: 2D numpy array (the output of Engine A)
        background_sigma: The SNR threshold required to trigger an extraction.
                          (e.g. 5.0 means the object must be 5x brighter than the noise floor)
                          
    Returns:
        objects: A structured numpy array of detections (includes 'x', 'y', 'a', 'b', 'flux').
                 These are the raw transient candidates sent to the Vetting Bouncer.
        globalrms: The global background RMS noise floor (used for significance calculation).
    """
    # Cast to float64 and ensure native byte order — SEP's C backend requires
    # native-endian arrays, and FITS files are big-endian by default.
    diff_data = np.ascontiguousarray(difference_image, dtype=np.float64)
    
    # Dynamically estimate the background RMS (noise floor) of the subtracted image
    bkg = sep.Background(diff_data)
    
    # Calculate the extraction threshold.
    # 20-sigma is intentionally aggressive. The primary target event class is supernovae,
    # which are extremely luminous and produce screaming detections on a difference image.
    # A high threshold prevents subtraction residuals near bright stars from flooding the
    # vetting pipeline with false positives, especially since there is no downstream ML classifier.
    thresh = background_sigma * bkg.globalrms
    
    # Extract contiguous blobs of pixels exceeding the threshold
    objects = sep.extract(diff_data - bkg.back(), thresh)
    
    return objects, bkg.globalrms
