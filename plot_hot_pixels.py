import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

filepath = r'S:\Jean\Interns\Vihaan\hot_pixels.fts'
print(f'Loading {filepath}...')

with fits.open(filepath) as hdul:
    data = hdul[0].data

# Calculate statistics
mean_val = np.mean(data)
std_val = np.std(data)
threshold = mean_val + (3 * std_val)

# Create a boolean mask for the hot pixels
hot_pixel_mask = data > threshold
num_hot_pixels = np.sum(hot_pixel_mask)
print(f"Plotting {num_hot_pixels} hot pixels...")

# To visualize this properly, we need to scale the dark background
# Using the 1st and 99th percentile gives a good contrast for the noise floor
vmin = np.percentile(data, 1)
vmax = np.percentile(data, 99)

# Normalize the data to 0.0 - 1.0 for RGB conversion
normalized_data = (data - vmin) / (vmax - vmin)
normalized_data = np.clip(normalized_data, 0, 1)

# Create an RGB image (H x W x 3)
# Start by making it purely greyscale
rgb_image = np.stack((normalized_data,)*3, axis=-1)

# Color the hot pixels RED!
# RGB: Red=1.0, Green=0.0, Blue=0.0
rgb_image[hot_pixel_mask] = [1.0, 0.0, 0.0]

# Plot the image
plt.figure(figsize=(12, 8), facecolor='black')
plt.imshow(rgb_image, origin='lower')
plt.title(f"Master Dark Frame ({num_hot_pixels} Hot Pixels colored in RED)", color='white', fontsize=16)
plt.axis('off')

# Save the plot so you can view it
output_path = r'C:\Users\calaf\.gemini\antigravity-ide\brain\d6c3d9bd-edb6-41aa-8217-bcdfa0ada2b7\hot_pixels_highlighted.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
print(f"Plot saved to: {output_path}")

# If running locally with a GUI, you can also uncomment the line below to pop open the window:
# plt.show()
