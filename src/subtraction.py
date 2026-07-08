import numpy as np
import sep
from scipy.signal import fftconvolve
import scipy.fft

def fit_optimal_kernel(target, reference, kernel_size=5):
    
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
    
    return K

def optimal_image_subtraction(target_image, reference_image, psf_kernel=None):
   
    if psf_kernel is None:
        # Calculate the dynamic atmospheric blur
        psf_kernel = fit_optimal_kernel(target_image, reference_image, kernel_size=5)
        
    # Artificially blur the reference image using Fast Fourier Transform convolution (Multi-Core)
    with scipy.fft.set_workers(-1):
        convolved_ref = fftconvolve(reference_image, psf_kernel, mode='same')
    
    # Subtract to isolate transients
    difference_image = target_image - convolved_ref
    
    return difference_image

def extract_sources_from_difference(difference_image, background_sigma=5.0):
    
    # sep requires contiguous memory in C byte order
    diff_data = np.ascontiguousarray(difference_image, dtype=np.float32)
    
    # Dynamically estimate the background RMS (noise floor) of the subtracted image
    bkg = sep.Background(diff_data)
    
    # Calculate the extraction threshold (5-sigma by default)
    thresh = background_sigma * bkg.globalrms
    
    # Extract contiguous blobs of pixels exceeding the threshold
    objects = sep.extract(diff_data - bkg.back(), thresh)
    
    return objects
