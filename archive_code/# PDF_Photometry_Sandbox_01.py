# PDF_Photometry_Sandbox_01

import matplotlib.pyplot as plt
from photutils.datasets import load_star_image

hdu = load_star_image()
plt.imshow(hdu.data, cmap='Greys', origin='lower',interpolation ='nearest')
plt.tight_layout()
plt.show()
