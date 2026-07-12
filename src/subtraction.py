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
            patch = reference[y_min+i : y_max+i : stride, x_min+j : x_max+j : stride]
            M[:, col] = patch.flatten()
            col += 1
            
    # --- SIGMA-CLIPPING: Exclude transient pixels from the kernel solve ---
    # We use the Median Absolute Deviation (MAD) as a robust noise estimator,
    # then reject any pixel row that deviates more than 5-sigma from the median.
    median_val = np.median(I_flat)
    mad = np.median(np.abs(I_flat - median_val))
    robust_std = 1.4826 * mad  # MAD-to-sigma conversion for Gaussian noise
    if robust_std > 0:
        good_mask = np.abs(I_flat - median_val) < 5.0 * robust_std
        # Only clip if we keep enough rows to solve the linear system
        if good_mask.sum() >= kernel_size ** 2:
            I_flat = I_flat[good_mask]
            M = M[good_mask, :]

    # Ridge penalty injected into the diagonal to prevent matrix singularity
    # (Ensures stability even if parts of the image are perfectly black)
    ridge = 1e-4 * np.eye(kernel_size**2)
    
    # Solve the linear system using the pseudo-inverse
    k_flat = np.linalg.solve(M.T @ M + ridge, M.T @ I_flat)
    K = k_flat.reshape((kernel_size, kernel_size))
    
    # Normalize the kernel to sum to 1 to preserve flux in the convolved reference.
    # An un-normalized kernel would globally scale the reference, leaving a systematic
    # DC offset across the entire difference image instead of pure noise + transients.
    if K.sum() != 0:
        K /= K.sum()
    
    return K

def optimal_image_subtraction(target_image, reference_image, psf_kernel=None):
    """
    Engine A: The Discovery Engine.
    
    This engine hunts for completely uncataloged objects (like a new supernova)
    that appear in empty space. It dynamically blurs the pristine reference image 
    to match the atmospheric distortion of the current frame, then subtracts them.
    
    Math: Difference = Target - (Reference ⊗ K)
    
    Returns:
        difference_image: A 2D array where static stars have been mathematically 
                          erased, leaving only pure noise and new transients.
    """
    if psf_kernel is None:
        # Calculate the dynamic atmospheric blur
        psf_kernel = fit_optimal_kernel(target_image, reference_image, kernel_size=5)
        
    # Artificially blur the reference image using Fast Fourier Transform convolution (Multi-Core)
    with scipy.fft.set_workers(-1):
        convolved_ref = fftconvolve(reference_image, psf_kernel, mode='same')
    
    # Subtract to isolate transients
    difference_image = target_image - convolved_ref
    
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
