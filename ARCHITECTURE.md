# Pipeline Architecture & File Manifest

The MIRA Piggyback Pipeline is modularized into 7 core mathematical and operational engines, plus a robust test suite.

## Core Modules

### 1. `orchestrator.py`
The "brain" of the daemon. It loops continuously, watching the camera spool directory for new FITS files. It manages the state machine (`BURN_IN` vs `MONITORING`), routes aligned images into Engine A and Engine B, aggregates their alert outputs, passes them through the Vetting Bouncers, and triggers the `AlertLogger`. It also catches alignment crashes to safely reboot the pipeline when the telescope slews.

### 2. `calibration.py`
Handles basic image processing. It aligns every incoming target image to the dynamic master reference image using `astroalign`. If the telescope has slewed, `astroalign` fails, and this module gracefully raises an `AlignmentError` to notify the orchestrator.

### 3. `starfinder.py`
The autonomous mapping engine. Because the pipeline operates without Internet or WCS coordinates, this module uses `DAOStarFinder` to dynamically locate the (X, Y) pixel centroids of all stars in the master reference frame. It enforces a `max_stars` cap (CPU protection) and a `saturation_level` mask (Photometry protection).

### 4. `photometry.py` (Engine B)
The light-curve engine. It tracks the brightness of every mapped star across time. It utilizes two triggers:
- **Z-score Trigger:** Catches sudden, 1-frame massive spikes (Flares).
- **Rolling Variance Trigger:** Catches slow, multi-frame wave fluctuations (Pulsators/Variables).
It aggressively truncates historical data to prevent memory leaks over long observing shifts.

### 5. `subtraction.py` (Engine A)
The transient discovery engine. It uses Alard-Lupton optimal kernel fitting (via the `sep` library) to mathematically match the Point Spread Function (PSF) and background noise of the reference image to the target image. It subtracts the two, erasing all stable stars and leaving only newly spawned transients (Supernovae). 

### 6. `vetting.py` (The Bouncer)
The ultimate defense against false positives.
- **Spatial Bouncer:** Rejects anomalies that are too sharp (Cosmic Rays) or too elongated (satellite trails).
- **Saturation Bouncer:** Instantly rejects any subtraction anomaly located near a saturated pixel (blooming/bleeding column artifact).
- **Temporal Bouncer:** Forces Engine A transients and Engine B variables to persist for 3 consecutive frames before allowing an alert to be logged.

### 7. `alert_logger.py`
The output engine. When a transient survives all vetting, this module crops a 200x200 pixel sub-image around the exact coordinate, plots red crosshairs over the target, and saves a timestamped PNG to disk for human review.

## The Test Suite
- `test_integration.py`: A massive 11-frame mock camera simulation that verifies all 7 modules work together, generating master references, catching injected flares, verifying injected supernovae, and gracefully rebooting during telescope slews.
- `test_saturation.py`: Verifies that the spatial masking successfully rejects saturated targets.
- `test_stress.py`: An extreme numerical stress test that pushes the CPU, RAM, and Disk I/O limits of the pipeline via Crowded Field, Long Shift, Massive Outbreak, and Thick Cloud simulations. 
- `test_subtraction.py`: Verifies the Alard-Lupton optimal kernel math.
- `generate_proof_plots.py`: Generates the visual matplotlib proofs of the pipeline's capabilities.
