"""
This example loads the tetra3 default database and solves for every image in the
tetra3/examples/data directory.
"""
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import tetra3
from photutils.aperture import CircularAperture
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import numpy as np
'''
# to generate a database from hip_main
t3 = tetra3.Tetra3()
t3.generate_database(max_fov=9, min_fov=3, star_max_magnitude=8, save_as='hip_database938')
exit()
'''
# Create instance and load the default database, built for 30 to 10 degree field of view.
# Pass `load_database=None` to not load a database, or to load your own.
t3 = tetra3.Tetra3(load_database='hip_database938', debug_folder=Path(__file__).parent)

# Path where images are
EXAMPLES_DIR = Path(__file__).parent
path = EXAMPLES_DIR / 'data'
for impath in path.glob('*'):
    print('Solving for image at: ' + str(impath))
    with Image.open(str(impath)) as img:
        centroids = tetra3.get_centroids_from_image(img, sigma=2, filtsize=31)
        if 1:
            plt.imshow(img)
            plt.scatter(centroids[:10, 1], centroids[:10, 0])
            plt.show()
            positions = np.transpose((centroids[:, 1], centroids[:, 0]))
            print(positions)
            apertures = CircularAperture(positions, r=4.0)
            norm = ImageNormalize(stretch=SqrtStretch())
            plt.imshow(img, cmap='Greys', origin='lower', norm=norm,
                       interpolation='nearest')
            apertures.plot(color='blue', lw=1.5, alpha=0.5)
            plt.show()
        # Here you can add e.g. `fov_estimate`/`fov_max_error` to improve speed or a
        # `distortion` range to search (default assumes undistorted image). There
        # are many optional returns, e.g. `return_matches` or `return_visual`. A core
        # aspect of the solution is centroiding (detecting the stars in the image).
        # You can use `return_images` to get a second return value to check the
        # centroiding process, the key `final_centroids` is especially useful.
        solution = t3.solve_from_image(img, fov_estimate=5, pattern_checking_stars=10) #distortion=[-.2, .1])
    print('Solution: ' + str(solution))
