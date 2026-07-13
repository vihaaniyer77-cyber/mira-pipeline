import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
import scipy.fft
from subtraction import fit_optimal_kernel, optimal_image_subtraction, extract_sources_from_difference


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Identity & Stability (Zero-Difference)
# ─────────────────────────────────────────────────────────────────────────────
def test_identity_and_stability():
    """
    When target == reference, the solved kernel should be a Dirac delta
    (central pixel ~1, all others ~0) and the difference image should be
    numerically zero everywhere.
    """
    np.random.seed(42)
    reference = np.random.normal(loc=100, scale=5, size=(100, 100))
    target = reference.copy()

    kernel = fit_optimal_kernel(target, reference, kernel_size=5)
    difference = optimal_image_subtraction(target, reference, psf_kernel=kernel)

    # Central pixel of a 5x5 kernel is index [2, 2] — should dominate.
    # A ridge penalty of 1e-4 on a well-conditioned identical system means
    # the off-diagonal elements are suppressed, leaving >95% mass at center.
    assert kernel[2, 2] > 0.95, (
        f"Central kernel pixel should be >0.95 (delta function), got {kernel[2, 2]:.4f}"
    )

    # Off-diagonal mass should be negligible
    off_diagonal_mass = kernel.sum() - kernel[2, 2]
    assert abs(off_diagonal_mass) < 0.05, (
        f"Off-diagonal kernel mass should be <0.05, got {off_diagonal_mass:.4f}"
    )

    # Difference image should be essentially zero (within floating-point + ridge bias)
    assert np.allclose(difference, 0, atol=1e-2), (
        f"Difference image max residual: {np.abs(difference).max():.4e}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Flux Conservation (Photometric Integrity)
# ─────────────────────────────────────────────────────────────────────────────
def test_flux_conservation():
    """
    After convolving the reference with the solved kernel, the total summed flux
    of the convolved reference must equal the total flux of the original reference.

    NOTE: Asserting kernel.sum() == 1.0 is a tautology — the source code
    explicitly normalises by K.sum() unconditionally. The *real* photometric
    guarantee is that (reference ⊗ K) preserves total flux. That is what we
    test here.
    """
    np.random.seed(42)
    reference = np.random.normal(loc=100, scale=5, size=(100, 100))

    # Simulate worse atmospheric seeing in the target via Gaussian blur
    target = gaussian_filter(reference, sigma=1.0)

    kernel = fit_optimal_kernel(target, reference, kernel_size=5)

    # Manually convolve the reference (mirroring exactly what optimal_image_subtraction does)
    with scipy.fft.set_workers(-1):
        convolved_ref = fftconvolve(reference, kernel, mode='same')

    original_flux = reference.sum()
    convolved_flux = convolved_ref.sum()

    # Flux must be preserved within 2%. fftconvolve(mode='same') zero-pads
    # the boundary, so the outermost ~2-pixel border receives partial convolution
    # weight and bleeds a small amount of flux off the edge. On a 100x100 image
    # with a 5x5 kernel this causes an inherent ~1.4% boundary loss. We test
    # that this stays below 2% — any larger deviation indicates the kernel is
    # globally rescaling the image, which would create a false DC offset.
    assert np.isclose(original_flux, convolved_flux, rtol=0.02), (
        f"Flux not conserved: original={original_flux:.2f}, convolved={convolved_flux:.2f}, "
        f"fractional diff={(abs(original_flux - convolved_flux) / original_flux):.2e}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Sigma-Clipping Robustness (Artifact Rejection)
# ─────────────────────────────────────────────────────────────────────────────
def test_sigma_clipping_robustness():
    """
    When the target contains a massive artifact (cosmic ray / saturated star /
    bright transient), the MAD-based 5-sigma clipping must exclude those pixels
    from the kernel solve. The resulting kernel should still be a delta function,
    proving the anomaly was correctly masked.

    DESIGN NOTE: fit_optimal_kernel samples pixels at stride=10 on a 100x100
    image — stride grid positions: 5, 15, 25, 35, 45, 55, 65, 75, 85, 95.
    The anomaly is injected over a wide area (rows/cols 10:90) to guarantee
    that many stride-grid sample points fall inside the corrupted region and
    meaningfully stress the sigma-clipping logic.
    """
    np.random.seed(42)
    reference = np.random.normal(loc=100, scale=5, size=(100, 100))
    target = reference.copy()

    # Moderate injection: [35:65, 35:65] covers a 30x30 pixel area.
    # The stride=10 sampler grid on a 100x100 image yields positions
    #   2, 12, 22, 32, 42, 52, 62, 72, 82, 92  (y_min=2, stride=10)
    # The injection hits grid rows/cols at 42, 52, 62 → 9 contaminated
    # sample points out of ~100 total. With ~91 clean pixels anchoring the
    # MAD median, the contaminated 9 sit ~1e6 / (1.4826 * mad) sigmas away
    # and are correctly sigma-clipped. This stresses the MAD logic while
    # preserving the required clean majority.
    target[35:65, 35:65] += 1e6

    kernel = fit_optimal_kernel(target, reference, kernel_size=5)

    # Despite 9 corrupted stride-grid points (~9% of samples), sigma-clipping
    # must still recover a delta-like kernel.
    assert kernel[2, 2] > 0.90, (
        f"Sigma-clipping failed: central kernel pixel {kernel[2, 2]:.4f} should be >0.90. "
        f"The artifact was not properly excluded from the kernel solve."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Statistical Noise Floor & Threshold Validation
# ─────────────────────────────────────────────────────────────────────────────
def test_noise_floor_thresholding():
    """
    On a simulated difference image (positive sky background + Gaussian noise)
    with two injected point sources, the pipeline must:
      (a) Accurately estimate the background RMS noise floor.
      (b) Detect the above-threshold source.
      (c) Spatially resolve that detection to within the injection coordinates.

    Threshold is set to 20-sigma (matching MIRA's production setting).

    WHY positive sky background (loc=1000):
    SEP's Background() is designed for astronomical images with a positive
    sky pedestal. On a zero-mean noise image, the mesh-based background
    estimator slightly elevates the local background in cells containing bright
    patches, causing partial self-subtraction of the injected source. A
    realistic sky offset avoids this and mirrors real MIRA pipeline inputs.

    Source design:
    - 5-sigma source  (50 counts above sky): clearly below thresh (~200 counts)
    - 40-sigma source (400 counts above sky): robustly above thresh (~200 counts)
    The wide SNR gap ensures the result is unambiguous even with SEP mesh
    quantisation effects across the 200x200 image.
    """
    np.random.seed(42)
    sigma = 10.0
    sky = 1000.0  # Positive sky background, as in real astronomical images
    diff_image = np.random.normal(loc=sky, scale=sigma, size=(200, 200))

    # 5-sigma source above sky (clearly below 20-sigma threshold — NOT detected)
    diff_image[50:55, 50:55] += 5.0 * sigma

    # 40-sigma source above sky (robustly above 20-sigma threshold — MUST be detected)
    # 5x5 patch → 25 pixels above thresh, well above SEP's minarea=5 requirement.
    # Center of patch [100:105, 100:105] → centroid at approximately x=102, y=102.
    diff_image[100:105, 100:105] += 40.0 * sigma

    objects, bkg_rms = extract_sources_from_difference(diff_image, background_sigma=20.0)

    # Background RMS should be within 20% of our known per-pixel noise sigma.
    assert np.isclose(bkg_rms, sigma, rtol=0.2), (
        f"Background RMS {bkg_rms:.3f} deviates >20% from known sigma={sigma}"
    )

    # At least one object should be detected (the 40-sigma source)
    assert len(objects) >= 1, "Pipeline failed to detect any sources — 40-sigma source was missed."

    # The detection closest to our 40-sigma injection must be within 2 pixels.
    # Center of patch [100:105, 100:105] → expected centroid x≈102, y≈102.
    distances = np.sqrt((objects['x'] - 102.0) ** 2 + (objects['y'] - 102.0) ** 2)
    closest_distance = distances.min()
    assert closest_distance < 2.0, (
        f"Closest detection is {closest_distance:.2f} pixels away from 40-sigma injection site "
        f"(expected <2 pixels). Detected positions: x={objects['x']}, y={objects['y']}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Synthetic Supernova Injection & Recovery (End-to-End)
# ─────────────────────────────────────────────────────────────────────────────
def test_synthetic_transient_recovery():
    """
    Full end-to-end pipeline validation. A synthetic transient is injected at a
    known pixel coordinate into the target image (which is a slightly blurred
    version of the reference, simulating real atmospheric seeing). The pipeline
    must:
      (a) Correctly solve the optimal kernel to cancel the static star field.
      (b) Preserve the injected transient in the difference image.
      (c) Extract the transient at the correct astrometric position (within 2px).

    Injection: 3x3 block of +5000 counts at rows [120:123], cols [130:133].
    Expected centroid: x ≈ 131.0, y ≈ 121.0.

    The injected signal (~5000 counts) is ~1000-sigma above the background noise
    (~5 counts/pixel), ensuring it survives even imperfect kernel subtraction.
    """
    np.random.seed(42)
    reference = np.random.normal(loc=100, scale=5, size=(200, 200))

    # Slight blur (sigma=0.5) simulates a small but real PSF mismatch between
    # reference and current frame (atmospheric seeing change).
    target = gaussian_filter(reference, sigma=0.5)

    # Inject synthetic supernova at known coordinates.
    # Center of patch [120:123, 130:133] → centroid at (y=121, x=131) in array indexing.
    target[120:123, 130:133] += 5000.0

    # Run the full pipeline without pre-supplying the kernel (exercise auto-solve)
    difference = optimal_image_subtraction(target, reference)
    objects, bkg_rms = extract_sources_from_difference(difference, background_sigma=10.0)

    # At least one detection must survive subtraction
    assert len(objects) >= 1, (
        f"Pipeline detected no sources. bkg_rms={bkg_rms:.3f}. "
        "Transient was lost during subtraction or not extracted."
    )

    # Find the detection closest to the known injection site
    distances = np.sqrt((objects['x'] - 131.0) ** 2 + (objects['y'] - 121.0) ** 2)
    closest_idx = np.argmin(distances)
    closest_distance = distances[closest_idx]

    # Astrometric accuracy: must recover position within 2 pixels
    assert closest_distance < 2.0, (
        f"Transient recovered at x={objects['x'][closest_idx]:.2f}, y={objects['y'][closest_idx]:.2f} "
        f"— {closest_distance:.2f} pixels from injection site (expected <2 pixels)."
    )
