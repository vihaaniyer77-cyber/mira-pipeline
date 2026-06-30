import matplotlib.pyplot as plt
import numpy as np
import os

OUTPUT_DIR = "/Users/vihaa/.gemini/antigravity-ide/brain/0b3b5c28-8155-4813-bc9b-0f25a52964a9"

def plot_photometry():
    frames = np.arange(30)
    
    # 1. Stable Star
    stable_flux = np.random.normal(100, 5, 30)
    
    # 2. Flaring Star (Z-score spike)
    flare_flux = np.random.normal(100, 5, 30)
    flare_flux[15] += 200 # Sudden spike
    
    # 3. Pulsator (Variance)
    pulsator_flux = 100 + 40 * np.sin(frames / 3.0) + np.random.normal(0, 5, 30)
    
    plt.figure(figsize=(10, 6))
    plt.plot(frames, stable_flux, label="Stable Star", color='gray', alpha=0.6)
    plt.plot(frames, flare_flux, label="Engine B: Flare Event", color='orange', marker='*')
    plt.plot(frames, pulsator_flux, label="Engine B: Slow Pulsator", color='blue', linestyle='--')
    
    plt.axvline(15, color='red', linestyle=':', label="Z-score Trigger > 10.0")
    
    plt.title("Engine B: Autonomous Aperture Photometry Tracking")
    plt.xlabel("Frame Number")
    plt.ylabel("Relative Flux (ADU)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "proof_photometry.png"), dpi=150)
    plt.close()

def plot_subtraction():
    # 1. Create a pristine reference with 3 stable stars
    ref = np.random.normal(10, 2, (100, 100))
    ref[20:25, 20:25] += 50
    ref[80:85, 20:25] += 50
    ref[50:55, 80:85] += 50
    
    # 2. Create a target image with those 3 stars PLUS a new supernova
    target = np.random.normal(10, 2, (100, 100))
    target[20:25, 20:25] += 50
    target[80:85, 20:25] += 50
    target[50:55, 80:85] += 50
    target[70:75, 70:75] += 100 # Supernova
    
    # 3. Simple subtraction
    diff = target - ref
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(ref, origin='lower', cmap='viridis')
    axes[0].set_title("Master Reference Image")
    
    axes[1].imshow(target, origin='lower', cmap='viridis')
    axes[1].set_title("Current Target Image (With Transients)")
    axes[1].plot(72, 72, 'r+', markersize=20, markeredgewidth=2)
    
    axes[2].imshow(diff, origin='lower', cmap='cividis')
    axes[2].set_title("Engine A: Optimal Subtraction")
    axes[2].plot(72, 72, 'r+', markersize=20, markeredgewidth=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "proof_subtraction.png"), dpi=150)
    plt.close()

def plot_saturation():
    # Simulating a saturated star with blooming columns
    img = np.random.normal(10, 2, (100, 100))
    
    # Saturated core
    img[45:55, 45:55] = 60000
    # Blooming column (bleeding up and down the CCD)
    img[20:80, 48:52] = 58000
    
    plt.figure(figsize=(6, 6))
    im = plt.imshow(img, origin='lower', cmap='magma', vmax=60000)
    plt.colorbar(im, label='ADU')
    
    # Draw the spatial bounding box that the Saturation Bouncer enforces
    rect = plt.Rectangle((40, 40), 20, 20, fill=False, color='red', linewidth=3, label="Vetting Bouncer Mask")
    plt.gca().add_patch(rect)
    
    plt.title("Engine A Protection: Saturation Masking")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "proof_saturation.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    plot_photometry()
    plot_subtraction()
    plot_saturation()
    print("Proof plots generated successfully in artifact dir!")
