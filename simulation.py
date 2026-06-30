import numpy as np
import matplotlib.pyplot as plt
from calibration import align_image, generate_master_reference
from subtraction import optimal_image_subtraction, extract_sources_from_difference
from photometry import PhotometryEngine
from vetting import spatial_profile_vetting
from scipy.ndimage import gaussian_filter
import os

def create_mock_star_field(shape=(100, 100), stars=None, extra_blur=0.0):
    """Creates a basic image with Gaussian stars"""
    image = np.random.normal(10, 2, shape) # Background noise
    if stars is None:
        return image
        
    for x, y, flux in stars:
        y_idx, x_idx = np.ogrid[0:shape[0], 0:shape[1]]
        r2 = (x_idx - x)**2 + (y_idx - y)**2
        sigma = 1.5
        gaussian = flux * np.exp(-r2 / (2 * sigma**2))
        image += gaussian
        
    if extra_blur > 0.0:
        image = gaussian_filter(image, sigma=extra_blur)
        
    return image

def run_simulation():
    print("Initializing pipeline engines for 4 variability types...")
    photometry_engine = PhotometryEngine(window_size=5, z_threshold=3.5, min_std=20.0)
    
    # 4 distinct types of sources
    # 0: No variability
    # 1: Pulsator
    # 2: Stellar Flare
    # 3: Supernova
    positions = [
        (20, 20), # No var
        (70, 30), # Pulsator
        (30, 80), # Flare
        (80, 70)  # Supernova
    ]
    
    ref_stars = [
        (positions[0][0], positions[0][1], 500), # Stable
        (positions[1][0], positions[1][1], 400), # Pulsator base
        (positions[2][0], positions[2][1], 600), # Flare base
        # Supernova not in reference yet
    ]
    
    print("Executing Burn-In Phase: Generating Dynamic Reference Template...")
    burn_in_frames = []
    for b in range(5):
        frame = create_mock_star_field(stars=ref_stars, extra_blur=0.1)
        # Add a simulated satellite streak to frame 2 to prove median stacking erases it
        if b == 2:
            frame[50:55, :] += 200
        burn_in_frames.append(frame)
        
    ref_image = generate_master_reference(burn_in_frames)
    print("Nightly Reference generated successfully from median stack of 5 frames.")
    
    num_frames = 30
    z_scores_history = {0: [], 1: [], 2: [], 3: []}
    fluxes_history = {0: [], 1: [], 2: [], 3: []}
    stds_history = {0: [], 1: [], 2: [], 3: []}
    diff_images_history = []
    
    print("Processing mock frames...")
    
    sn_peak_frame = None
    
    for i in range(num_frames):
        # 0: No variability
        flux_no_var = 500
        
        # 1: Pulsator (sinusoidal)
        flux_pulsator = 400 + 100 * np.sin(i * 2 * np.pi / 10)
        
        # 2: Stellar Flare (fast rise at frame 10, exp decay)
        if i < 10:
            flux_flare = 600
        elif i == 10:
            flux_flare = 1500 # spike
        else:
            flux_flare = 600 + (1500 - 600) * np.exp(-(i - 10) / 2.0)
            
        # 3: Supernova (slow rise starting at frame 5)
        if i < 5:
            flux_sn = 0
        else:
            # logistic-like slow rise
            flux_sn = 800 / (1 + np.exp(-0.5 * (i - 15)))
            
        current_stars = [
            (positions[0][0], positions[0][1], flux_no_var),
            (positions[1][0], positions[1][1], flux_pulsator),
            (positions[2][0], positions[2][1], flux_flare),
            (positions[3][0], positions[3][1], flux_sn)
        ]
        
        # Target image has worse atmospheric seeing (extra_blur > 0)
        # to prove the OIS dynamic kernel fit can handle matching PSFs
        target_image = create_mock_star_field(stars=current_stars, extra_blur=0.6)
        aligned = align_image(target_image, ref_image)
        
        fluxes = photometry_engine.perform_aperture_photometry(aligned, positions)
        
        for j, f in enumerate(fluxes):
            fluxes_history[j].append(f)
            
        z_scores, stds, z_alerts, var_alerts = photometry_engine.update_light_curves(fluxes)
        for j, z in enumerate(z_scores):
            z_scores_history[j].append(z)
        for j, std in enumerate(stds):
            stds_history[j].append(std)
            
        # Subtraction Engine
        difference_image = optimal_image_subtraction(aligned, ref_image)
        diff_images_history.append(difference_image)
        sources = extract_sources_from_difference(difference_image)
        
        if i == 20:
            sn_peak_frame = difference_image
            
    # ---------------------------------------------------------
    # Generate 12 Individual Plots
    # ---------------------------------------------------------
    print("Generating 12 individual plots...")
    names = {
        0: "No_Variability",
        1: "Pulsator",
        2: "Stellar_Flare",
        3: "Supernova"
    }
    
    # We will grab specific difference images for each to show them best
    # For Flare, it peaks at frame 10. Supernova at 20.
    peak_frames = {
        0: 20, # No Var
        1: 20, # Pulsator
        2: 10, # Stellar Flare
        3: 20  # Supernova
    }
    
    for obj_id in range(4):
        # 1. Light Curve
        plt.figure(figsize=(8, 5))
        plt.plot(fluxes_history[obj_id], label=f"{names[obj_id]} Flux", color='blue', linewidth=2)
        plt.title(f"Light Curve: {names[obj_id]}")
        plt.xlabel("Frame Number")
        plt.ylabel("Extracted Flux")
        plt.grid(True)
        plt.savefig(f"light_curve_{names[obj_id]}.png")
        plt.close()
        
        # 2. Z-Score
        plt.figure(figsize=(8, 5))
        plt.plot(z_scores_history[obj_id], label=f"{names[obj_id]} Z-Score", color='red', linewidth=2)
        plt.axhline(3.5, color='black', linestyle=':', label="Alert Threshold (+3.5σ)")
        plt.axhline(-3.5, color='black', linestyle=':', label="Alert Threshold (-3.5σ)")
        plt.title(f"Z-Score (Flares): {names[obj_id]}")
        plt.xlabel("Frame Number")
        plt.ylabel("Z-Score")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.savefig(f"z_score_{names[obj_id]}.png")
        plt.close()
        
        # 2b. Variance Plot
        var_threshold = 20.0 * 3.0 # min_std * var_threshold_multiplier
        plt.figure(figsize=(8, 5))
        plt.plot(stds_history[obj_id], label=f"{names[obj_id]} Rolling Std Dev", color='orange', linewidth=2)
        plt.axhline(var_threshold, color='black', linestyle=':', label="Variable Alert Threshold")
        plt.title(f"Variance (Pulsators): {names[obj_id]}")
        plt.xlabel("Frame Number")
        plt.ylabel("Standard Deviation")
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.savefig(f"variance_{names[obj_id]}.png")
        plt.close()
        
        # 3. Difference Image (Cropped 40x40 around the object at its peak frame)
        frame_idx = peak_frames[obj_id]
        diff_frame = diff_images_history[frame_idx]
        px, py = positions[obj_id]
        
        # Calculate crop bounds (ensure we don't go out of bounds)
        size = 20
        y_min, y_max = max(0, py - size), min(100, py + size)
        x_min, x_max = max(0, px - size), min(100, px + size)
        
        cropped_diff = diff_frame[y_min:y_max, x_min:x_max]
        
        plt.figure(figsize=(6, 6))
        plt.imshow(cropped_diff, cmap='viridis', origin='lower')
        plt.colorbar(label='Difference Flux')
        plt.title(f"Difference Image: {names[obj_id]}\n(Cropped around object, Frame {frame_idx})")
        plt.savefig(f"difference_image_{names[obj_id]}.png")
        plt.close()
        
    print("Simulation complete. 12 plots saved.")

if __name__ == "__main__":
    run_simulation()
