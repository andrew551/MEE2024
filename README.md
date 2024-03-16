# MEE2024
Modern Eddington Experiment codebase

- To run the Python source code, install the most recent version of Python from python.org.

- During installation under Windows, make sure to check the box to include the PATH.

- For a Windows install, double click the _windows_setup_ batch file to install the needed Python libraries.

- If you are installing an update of this software, you can double click on the _window_update_ batch file to make sure you are using the most recent Python libraries. 

- It is important that the PIP package manager itself is up to date for some of the libraries to install correctly.

A simpler way to run this software is to use the Windows executable file, which can be found under **Releases**.
In this case, no Python environment is required. Simply double-click on the executable file or call it using a command line entry.


### **Usage**

For stacking, run the Python file _MEE2024Stacker_ or the Windows executable file.
Select the images to be stacked (the "Light frames").
Dark frame(s) and Flat frame(s) can also be selected, if desired.

It is recommended to choose an _Output folder_ or else the output files will be written to the same folder which contains the image data.

Select "Show graphics" if you want to view the intermediate graphical analysis (optional).
Choose the number of bright stars to be identified in the stacked image (the default of 100 is a reasonable choice).

The output FITS stacked image is by default resized to 16-bit (the same as the input). A second 32-bit floating point (FP) FITS file can also be saved (optional).
The program does its calculations in 32-bit FP to preserve accuracy. The 32-bit FP file will be twice the size of the 16-bit file.

The Dark and Flat stacked images can also be saved (optional); the format is 32-bit FP FITS. These stacks can be used in subsequent processing of Light frames.

"Remove big bright object" is useful when images contain the Sun or Moon. It can be kept enabled for star fields with no Sun or Moon.
The _blob_radius_extra_ parameter determines the extra exclusion zone outside the saturated region. The "extra" distance is measured in pixels.
The _centroid_gap_blob_ parameter determines which centroids outside the extra exclusion zone should be ignored. The "gap" distance is measured in pixels.
Neither parameter is particularly sensitive. The purpose of this function is to limit the centroid search to areas away from the Moon and the solar corona.

"Sensitive stacking mode" should only be use if images contain the Sun or Moon or are taken on a bright sky (e.g. twilight).
For dark-sky star fields, this mode will take too long and is not recommended.

"Use sensitive mode on stacked result" can be left on for most images which require accurate centroid finding, but the sensitivity parameters should adjusted accordingly.
A lower _sigma_thresh_ will increase the sensitivity (between 4 and 7 are typical values).
A smaller _min_area_ will mean centroids of smaller pixel size will be found (between 1 and 4 are typical values).
A higher _sigma_subtract_ will increase the background cutoff, thereby eliminating more noise and reducing the number of centroids found (between 0 and _sigma_thresh_ -2 are typical values). For good dark-sky images, (5.0, 4, 3.0) are reasonable values.

"Remove centroids near edges" will remove extraneous centroids associated with edge effects such as the Moon or the solar corona. It can be left on, just like "Remove big bright object".

The file called _MEE_config.txt_ stores the program parameters, including the input and output directories.
It is automatically updated each time the program is run, and can also be manually edited.
