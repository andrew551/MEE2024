#
# AstroPy stats 
from astropy.stats import sigma_clipped_stats
from photutils.datasets import load_star_image
hdu = load_star_image()
data = hdu.data[200:800,200:1200]
mean, median, std = sigma_clipped_stats(data, sigma=3.0)
print(mean, median,std)

# DAOStarFinder
from photutils.detection import DAOStarFinder
daofind = DAOStarFinder(fwhm=3.0, threshold=5.0*std)
sources = daofind(data-median)
for col in sources.colnames:
    sources[col].info.format = '%.8g'
print(sources)

# mark the images
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.aperture import CircularAperture
positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
print(positions)
apertures = CircularAperture(positions, r=4.0)
norm = ImageNormalize(stretch=SqrtStretch())
plt.imshow(data, cmap='Greys', origin='lower', norm=norm,
           interpolation='nearest')
apertures.plot(color='blue', lw=1.5, alpha=0.5)
plt.show()