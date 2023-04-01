
# Q1

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from PIL import Image
import warnings
pd.options.display.float_format = '{:.5f}'.format
pd.set_option('display.max_columns', 500)
warnings.filterwarnings('ignore')

### a

# Load the image using PIL
img = Image.open("Cliff_beach_BOEM_gov.jpg")

# Get the dimensions of the image
width, height = img.size
print("Image width: {}, height: {}, that is number of pixels.".format(width, height))

### b

# Convert the image to a numpy array
img_arr = np.array(img)

# Normalize the image by subtracting the mean and dividing by the standard deviation
img_arr_normalized = (img_arr - np.mean(img_arr)) / np.std(img_arr)

# Clip the values in the array to the range [0, 1]
img_arr_normalized = np.clip(img_arr_normalized, 0, 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(img_arr)
ax1.set_title("Original Image")
ax2.imshow(img_arr_normalized)
ax2.set_title("Normalized Image")
plt.show()
### c

# Downscale the normalized image to reduce the computation time
scale_factor = 0.2
img_arr_normalized_downscaled = img_arr_normalized[::int(1/scale_factor), ::int(1/scale_factor), :]

# Perform SVD decomposition and obtain the singular values
U, s, Vt = np.linalg.svd(img_arr_normalized_downscaled, full_matrices=False)
s_values = s / s[0] # Normalize the singular values by the largest one

# Plot the singular values and the threshold
threshold = 0.0001 * s_values[0] # 0.01% of the largest singular value
num_singular_values = np.sum(s_values > threshold)
plt.plot(s_values)
plt.axhline(y=threshold, color='r', linestyle='--')
plt.title(f"Singular Values ({num_singular_values} values > {threshold:.4f})")
plt.show()


