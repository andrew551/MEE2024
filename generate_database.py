"""
@author: Andrew Smith
Version 4 January 2024
"""

import tetra3
t3 = tetra3.Tetra3()
t3.generate_database(max_fov=9, min_fov=1, star_max_magnitude=9, save_as='tyc_dbase4', star_catalog='tyc_main')
#t3.generate_database(max_fov=9, min_fov=3, star_max_magnitude=8, save_as='hip_dbase938', star_catalog='hip_main')
