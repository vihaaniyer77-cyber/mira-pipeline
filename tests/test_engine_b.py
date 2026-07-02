import os
import numpy as np
from astropy.io import fits
import sys

# Add src to path priority
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from photometry import PhotometryEngine

def generate_2d_gaussian(shape, amplitude, x0, y0, sigma_x, sigma_y):
    x = np.arange(0, shape[1], 1)
    y = np.arange(0, shape[0], 1)
    x, y = np.meshgrid(x, y)
    g = amplitude * np.exp(-((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2)))
    return g

def mag_to_flux(mag, zp=25.0):
    """Convert magnitude to flux."""
    return 10**((zp - mag) / 2.5)

def inject_signal_to_image(base_image, mag, x0, y0, sigma=2.0):
    """Injects a Gaussian star with the specified magnitude into the image."""
    flux = mag_to_flux(mag)
    # Estimate amplitude such that the total sum equals flux
    # integral of 2D Gaussian is 2 * pi * amplitude * sigma_x * sigma_y
    amplitude = flux / (2 * np.pi * sigma**2)
    
    star_profile = generate_2d_gaussian(base_image.shape, amplitude, x0, y0, sigma, sigma)
    
    # Add Poisson noise to the injected signal
    noisy_star = np.random.poisson(np.clip(star_profile, 0, None))
    return base_image + noisy_star

def get_flare_magnitude(step, baseline_mag):
    """Astrophysically valid M-Dwarf Flare model (fast rise, exp decay)."""
    t_hours = (step * 3.6) / 60.0  
    flare_start_hours = 0.5  
    peak_hours = flare_start_hours + (6.0 / 60.0)  
    
    if t_hours < flare_start_hours:
        flare_delta = 0.0
    elif flare_start_hours <= t_hours <= peak_hours:
        fractional_rise = (t_hours - flare_start_hours) / (peak_hours - flare_start_hours)
        flare_delta = -2.0 * np.sqrt(fractional_rise) # max 2 mag drop
    else:
        dt = t_hours - peak_hours
        flare_delta = -2.0 * np.exp(-dt / 1.0)
        
    return baseline_mag + flare_delta

def get_pulsator_magnitude(step, baseline_mag):
    """Delta Scuti rapid pulsator model."""
    t_hours = (step * 3.0) / 60.0  
    period_hours = 1.5  
    fundamental = 0.20 * np.sin(2 * np.pi * t_hours / period_hours)
    overtone = 0.04 * np.sin(2 * np.pi * t_hours / (period_hours / 2.0))
    return baseline_mag + fundamental + overtone

def run_test():
    # Load a real background image to act as the base
    # We will use a small crop of it to keep it fast
    fits_path = r"S:\Jean\Interns\Vihaan\reduced_frames\reduced_20260508-KIC11674677-0001g.fit"
    print(f"Loading base image: {fits_path}")
    try:
        with fits.open(fits_path) as hdul:
            full_image = hdul[0].data
        # Take a 200x200 crop from the center
        cy, cx = full_image.shape[0]//2, full_image.shape[1]//2
        base_image = full_image[cy-100:cy+100, cx-100:cx+100]
    except Exception as e:
        print(f"Could not load image: {e}")
        return

    # Let's test a baseline magnitude of 20.0
    baseline_mag = 20.0
    target_pos = (100, 100) # Center of the crop
    positions = [target_pos]

    TOTAL_STEPS = 50
    
    print("\n--- Test 1: Flat Star (No Variability) ---")
    engine = PhotometryEngine()
    flat_z_alerts = []
    flat_var_alerts = []
    for step in range(TOTAL_STEPS):
        mag = baseline_mag
        # Inject star
        img = inject_signal_to_image(base_image, mag, target_pos[0], target_pos[1])
        fluxes = engine.perform_aperture_photometry(img, positions)
        z_scores, stds, z_alerts, var_alerts = engine.update_light_curves(fluxes)
        flat_z_alerts.extend(z_alerts)
        flat_var_alerts.extend(var_alerts)
        if step % 10 == 0:
            print(f"Flat Step {step:02d} | Std: {stds[0] if stds else 0:.2f}")
    print(f"Total Flare Alerts: {len(flat_z_alerts)} (Expected 0)")
    print(f"Total Pulsator Alerts: {len(flat_var_alerts)} (Expected 0)")


    print("\n--- Test 2: M-Dwarf Flare ---")
    engine = PhotometryEngine()
    flare_z_alerts = []
    for step in range(TOTAL_STEPS):
        mag = get_flare_magnitude(step, baseline_mag)
        img = inject_signal_to_image(base_image, mag, target_pos[0], target_pos[1])
        fluxes = engine.perform_aperture_photometry(img, positions)
        z_scores, stds, z_alerts, var_alerts = engine.update_light_curves(fluxes)
        flare_z_alerts.extend(z_alerts)
        
        # Print stats for flare
        if step % 5 == 0 or len(z_alerts) > 0:
            print(f"Step {step:02d} | Mag: {mag:.2f} | Z-score: {z_scores[0]:.2f} | Alert: {len(z_alerts)>0}")

    print(f"Total Flare Alerts: {len(flare_z_alerts)} (Expected > 0 during peak)")


    print("\n--- Test 3: Delta Scuti Pulsator ---")
    engine = PhotometryEngine()
    puls_var_alerts = []
    for step in range(TOTAL_STEPS):
        mag = get_pulsator_magnitude(step, baseline_mag)
        img = inject_signal_to_image(base_image, mag, target_pos[0], target_pos[1])
        fluxes = engine.perform_aperture_photometry(img, positions)
        z_scores, stds, z_alerts, var_alerts = engine.update_light_curves(fluxes)
        puls_var_alerts.extend(var_alerts)
        
        if step % 5 == 0 or len(var_alerts) > 0:
            print(f"Step {step:02d} | Mag: {mag:.2f} | Std: {stds[0] if stds else 0:.2f} | Var Alert: {len(var_alerts)>0}")

    print(f"Total Pulsator Alerts: {len(puls_var_alerts)} (Expected > 0)")


if __name__ == "__main__":
    run_test()
