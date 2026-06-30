import numpy as np
from scipy.signal import fftconvolve

shape=(15,15)
image = np.zeros(shape)
y, x = np.ogrid[0:shape[0], 0:shape[1]]
r2 = (x-7)**2 + (y-7)**2
image += 500 * np.exp(-r2 / (2 * 1.5**2))

psf_kernel = np.array([[0.05, 0.1, 0.05], [0.1,  0.4, 0.1], [0.05, 0.1, 0.05]], dtype=float)
convolved = fftconvolve(image, psf_kernel, mode='same')

diff = image - convolved
print("Original center:", image[7,7])
print("Convolved center:", convolved[7,7])
print("Difference center (spike):", diff[7,7])
print("Difference ring:", diff[7,6])
