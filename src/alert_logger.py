import os
import csv
import datetime
import numpy as np

class AlertLogger:
  
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
                writer.writerow(["Timestamp", "Engine", "Significance", "X_Pixel", "Y_Pixel", "RA", "Dec", "Image_File"])

    def log_alert(self, engine_name, x, y, full_image, crop_size=50, wcs=None, significance="Unknown"):
        
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
                    f.write(f"Significance: {significance}\n")
                    f.write(f"RA (deg): {ra_str}\n")
                    f.write(f"Dec (deg): {dec_str}\n")
                    f.write(f"Pixel: X:{x}, Y:{y}\n")
            except Exception as e:
                print(f"Warning: Failed to convert pixels to RA/Dec: {e}")
               
        # 5. Append to the localized CSV database
        try:
            with open(self.csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, engine_name, significance, x, y, ra_str, dec_str, txt_filename])
        except Exception as e:
            print(f"Warning: Failed to write to CSV database: {e}")
           
        # 6. Console Notification
        print(f"DISCOVERY LOGGED [{engine_name} | {significance}] at RA:{ra_str}, Dec:{dec_str} (X:{x}, Y:{y}). Saved to {txt_filename}")
