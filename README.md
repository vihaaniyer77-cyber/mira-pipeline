# MIRA Autonomous Piggyback Pipeline

**MIRA (Monterey Institute for Research in Astronomy) Piggyback Pipeline** is a fully autonomous, blind-survey transient detection system designed to operate on a secondary, wide-field telescope (e.g., a 14-inch scope) that is physically mounted ("piggybacking") on a primary telescope (e.g., a 36-inch scope).

## The Problem
When the primary 36-inch telescope is slewing across the sky to target specific exoplanets or variable stars, the piggybacked 14-inch telescope is dragged along with it, staring blindly at random patches of the sky. 

## The Solution
Instead of letting the 14-inch gather dust, this pipeline turns it into an autonomous transient discovery engine. The pipeline runs as a daemon, constantly reading FITS files written by the 14-inch camera, and uses advanced mathematics to hunt for Supernovae, Exoplanet flares, and variable stars in real-time.

## Key Capabilities
- **WCS-Independent:** It operates entirely without World Coordinate System (WCS) headers or external catalogs (like Gaia). It autonomously maps the star field dynamically.
- **Dual-Engine Architecture:**
  - **Engine A (Image Subtraction):** Uses Alard-Lupton optimal kernel convolution to align and subtract images, erasing the static universe to reveal newly spawned transients (Supernovae, Novae, Asteroids).
  - **Engine B (Aperture Photometry):** Tracks the light curves of thousands of stable stars simultaneously, using Z-scores to instantly flag 1-frame Flares, and Rolling Variance to flag slow Pulsators.
- **The Bouncer (Vetting):** A strict vetting system that rejects cosmic rays via spatial FWHM checks, rejects subtraction anomalies via saturation masking (55,000 ADU threshold), and verifies true transients via 3-frame temporal persistence.
- **Hardware Hardened:** Built to survive remote, air-gapped observatories. Features memory-leak truncation, I/O race-condition locks, CPU extraction caps, and graceful cache flushing when the primary telescope unexpectedly slews.

## Usage
Activate the environment and run the orchestrator daemon pointing to your camera's spool directory:

```bash
python run_pipeline.py /path/to/camera/spool
```

Whenever the pipeline detects a verified transient, it will automatically crop a sub-image, draw crosshairs over the target, and save an Alert PNG to the directory for human review.
