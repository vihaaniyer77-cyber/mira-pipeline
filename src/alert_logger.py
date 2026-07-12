import os
import csv
import datetime
import numpy as np

class AlertLogger:
    """
    Because the 14-inch telescope computer is air-gapped and lacks a WCS grid,
    this module acts as the localized 'Database and Notification System'.
   
    When a transient survives all vetting, this logger writes the exact pixel
    coordinates to a CSV file and saves a text file of the event to a
    local folder for manual morning review by the astronomer.
    """
    def __init__(self, output_dir="pipeline_discoveries"):
        self.output_dir = output_dir
        self.csv_path = os.path.join(self.output_dir, "discoveries.csv")
       
        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
           
        # Initialize the CSV with headers if it doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Engine", "X_Pixel", "Y_Pixel", "RA", "Dec", "Image_File"])

    def log_alert(self, engine_name, x, y, full_image, crop_size=50, wcs=None):
        """
        Logs a confirmed transient alert to the CSV and saves a text file.
       
        Args:
            engine_name: 'Engine A (New)' or 'Engine B (Flare/Variable)'
            x, y: The exact pixel coordinates of the transient
            full_image: The 2D numpy array of the current camera frame (or difference image)
            crop_size: The width/height of the PNG cutout in pixels
            wcs: Optional Astropy WCS object for sky coordinates
        """
        # Ensure integers for indexing
        x, y = int(round(x)), int(round(y))
       
        # 1. Generate a unique timestamp filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        engine_short = engine_name.split()[1].replace("(", "").replace(")", "") if " " in engine_name else engine_name
        txt_filename = f"{timestamp}_{engine_short}_X{x}_Y{y}.txt"
        txt_filepath = os.path.join(self.output_dir, txt_filename)
       
        # 2. Handle Astrometry (WCS Transformation)
        ra_str = "Unknown"
        dec_str = "Unknown"
        if wcs is not None:
            try:
                sky_coord = wcs.pixel_to_world(x, y)
                ra_str = f"{sky_coord.ra.deg:.5f}"
                dec_str = f"{sky_coord.dec.deg:.5f}"
               
                # Write an explicit text file so the astronomer can copy-paste RA/Dec
                with open(txt_filepath, 'w') as f:
                    f.write(f"--- MIRA PIPELINE ALERT ---\n")
                    f.write(f"Type: {engine_name}\n")
                    f.write(f"RA (deg): {ra_str}\n")
                    f.write(f"Dec (deg): {dec_str}\n")
                    f.write(f"Pixel: X:{x}, Y:{y}\n")
            except Exception as e:
                print(f"Warning: Failed to convert pixels to RA/Dec: {e}")
               
        # 5. Append to the localized CSV database
        try:
            with open(self.csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, engine_name, x, y, ra_str, dec_str, txt_filename])
        except Exception as e:
            print(f"Warning: Failed to write to CSV database: {e}")
           
        # 6. Console Notification
        print(f"DISCOVERY LOGGED [{engine_name}] at RA:{ra_str}, Dec:{dec_str} (X:{x}, Y:{y}). Saved to {txt_filename}")
