# MEE2024
Modern Eddington Experiment codebase

- To run the Python source code, install the most recent version of Python from python.org.

- During installation under Windows, make sure to check the box to include the PATH.

- For a Windows install, double click the _windows_setup_ batch file to install the needed Python libraries.

- If you are installing an update of this software, you can double click on the _window_update_ batch file to make sure you are using the most recent Python libraries. 

- It is important that the PIP package manager itself is up to date for some of the libraries to install correctly.

In order to use the plate solving function, you also need a database of stars in addition to the program files.
However, plate solving is an optional function. A selection of prepared databases can be found in the **Releases** section.

Plate solving is done using the Tetra3 libary. For information about Tetra3, see:
https://tetra3.readthedocs.io/en/latest/api_docs.html.

A simpler way to run this software is to use the Windows executable file, which can be found under **Releases**.
In this case, no Python environment is required. Simply double-click on the executable file or call it using a command line entry.


### **Usage**

For stacking, run the Python file _MEE2024Stacker_ or the Windows executable file.
Select the images to be stacked (the "Light frames").
Dark frame(s) and Flat frame(s) can also be selected, if desired.
Choose the plate solving database (optional).

It is recommended to choose an _Output folder_ or else the output files will be written to the same folder which contains the image data.

Select "Show graphics" if you want to view the intermediate graphical analysis (optional).
Choose the number of bright stars to be identified in the stacked image (around 100 is a reasonable choice).

The output FITS stacked image is by default resized to 16-bit (the same as the input). A second 32-bit floating point (FP) FITS file can also be saved (optional).
The program does its calculations in 32-bit FP to preserve accuracy. The 32-bit FP file will be twice the size of the 16-bit file.

The stack of Darks and Flat can also be saved (optional). These are saved as 32-bit FP FITS files.

"Remove big bright object" is useful when images contain the Sun or Moon. It can be kept enabled for star fields with no Sun or Moon.

"Sensitive centroid finder mode" should only be use if images contain the Sun or Moon or are taken on a bright sky (e.g. twilight).
For dark-sky star fields, this mode will take too long and is unnecessary (i.e. it is not recommended).

The plate solver (if used) will identify around 8 to 10 of the brightest stars.
Depending on the database used (Hipparcos or Tycho), you can use a search tool to find the details about the identified stars.

See, for example, https://hipparcos-tools.cosmos.esa.int/HIPcatalogueSearch.html.

HIP identifiers are in the form _aaaaa_.

TYC identifiers are in the form _aaaa-bbbbb-1_.

The file _generate_database_ can be used to create new plate solving databases (with different fields of view and limiting magnitudes).
Both the HIP and TYC star catalogues can be used. A large database can take many hours to generate.

The program will generate a file _MEE_config.txt_ which contains the last used parameters (e.g. directories used).
If this file does not initially exist, it will be created (an error message will be displayed in the console noting that the file does not yet exist).

