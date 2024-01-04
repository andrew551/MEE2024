"""The goal of this was to create a program that detected stars in a fits image and 
drew a circle around them, and then store their pixel coord and intensity in a dictionary.
The issue that derailed this goal was the fact that the pixel coords are a float val and not
an int. I have commented out those parts of the code but you should be able to upload a fits
image and it will plot the stars as plt, circle them, print out the pixel coords and their intensities
in the terminal"""

import matplotlib.pyplot as plt
from astropy.io import fits
from skimage.feature import blob_log
from photutils.datasets import load_star_image

# Set the marker size
marker_size = 20

# Set the font size for the labprint(skimage.___version__)els
font_size = 12

# Set the alpha value for the markers
alpha = 0.2

# Set the color for the markers
marker_color = 'yellow'


# Load the FITS image
#hdulist = fits.open('ref_test.fts')
#hdulist = fits.open('D:\images-from-2023-09-30-observing-session\ProcessedImages\ZenithCenteredStack_20x3s.fit')
hdulist = [load_star_image()]
image_data = hdulist[0].data

# Set the minimum and maximum sigma values for the blobs
min_sigma = 2
max_sigma = 20

# Set the threshold for detecting blobs
threshold = 0.01

# Detect blobs in the image (which may correspond to stars)
blobs = blob_log(image_data, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)

# Extract the coordinates of the blobs
coordinates = blobs[:, :2]

#print pixel coordinates of found blobs/centroids
print(f"List of found Centriods Pixel Coordinates \n{coordinates}")
print(f"Pixel Value of each found centroid: \n{blobs}")

#create a dictionary of pixel coords and pixel vals using 
#a dictionary comprehension
#centroid_dictionary = {((int(x),int(y))): image_data[int(y),int(x)] for x, y in coordinates}
#print(f"Dictionary of Centroids\nkey(x,y):Value(pixel value)\n{centroid_dictionary}")

# Plot the image and mark the locations of the blobs
plt.imshow(image_data, cmap='gray')#cmap='hsv')
#plt.scatter(coordinates[:, 1], coordinates[:, 0], marker='o', color='red',alpha=0.2)
plt.scatter(coordinates[:, 1], coordinates[:, 0], marker='+', color='red',alpha=0.2)

# Iterate over the coordinates and add a label for each marker
#for i, coord in enumerate(coordinates):
    #x, y = coord
    #plt.scatter(x, y, marker='o', color=marker_color, alpha=alpha, s=marker_size)
    #plt.text(x+marker_size, y+marker_size, i+1, fontsize=font_size)

plt.show()
