"""Update: This program takes a fits image, and searches through pixels and finds the highest value pixels
in the array. It stores both the px coord and px values as a numpy array. Gathered data is then stored in a
dictionary with the px coords converted into a tuple of str type and the px value is stored as the value of
its cooresponding key. The gathered data is then plotted and indicated with a yellow circle showing each of the 
found blobs that should be stars"""

import matplotlib.pyplot as plt
from astropy.io import fits
from skimage.feature import blob_log

#Plot Constants
# Set the marker size
marker_type = "o"
# Set the alpha value for the markers
alpha_val = 0.2
# Set the color for the markers
marker_color = 'yellow'


# Load the FITS image
hdulist = fits.open('test_ref.fts')
image_data = hdulist[0].data

# Set the minimum and maximum sigma values for the blobs
min_sigma = 1
max_sigma = 5

# Set the threshold for detecting blobs
threshold = 0.01

# Detect blobs in the image (which may correspond to stars)
blobs = blob_log(image_data, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)

# Extract the coordinates of the blobs
coordinates = blobs[:, :2]

#print pixel coordinates of found blobs/centroids
print("##########################################")
print("Data before compiled to a dictionary")
print(f"List of found Centriods Pixel Coordinates \n{coordinates}")
print(f"Pixel Value of each found centroid: \n{blobs}")
print("##########################################")

#Convert px coords to tuple of type str and set as key in dictionary, loop and place found blob px value as value of matching key.
dict_blobs_found = {str(tuple(coordinates[i])): blobs[i] for i in range(len(coordinates))}

print("##########################################")
print(f"Dictionary of blobs found with px coord as key, and px val as dict value:\n{dict_blobs_found}")
print("##########################################")

# Plot the image and mark the locations of the blobs
plt.imshow(image_data, cmap='hsv')
plt.scatter(coordinates[:, 1], coordinates[:, 0], marker=marker_type, color=marker_color,alpha=alpha_val)
plt.show()
