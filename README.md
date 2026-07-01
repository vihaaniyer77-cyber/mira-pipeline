# MIRA Autonomous Piggyback Pipeline

**MIRA (Monterey Institute for Research in Astronomy)  Pipeline** is a fully autonomous, blind-survey transient detection system designed to operate on a secondary, wide-field telescope (e.g., a 14-inch scope) that is physically mounted on a primary telescope (e.g., a 36-inch scope).

## The Problem
When the primary 36-inch telescope is slewing across the sky to target variable stars, the piggybacked 14-inch telescope is dragged along with it, staring blindly at random patches of the sky. 

## The Solution
Instead of letting the 14-inch gather dust, this pipeline turns it into an autonomous transient discovery engine. The pipeline runs as a daemon, constantly reading FITS files written by the 14-inch camera, and uses advanced mathematics to hunt for Supernovae, Stellar flares, and variable stars in real-time.


Whenever the pipeline detects a verified transient, it will automatically crop a sub-image, draw crosshairs over the target, and save an Alert PNG to the directory for human review.
