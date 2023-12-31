# MEE2024
Modern Eddington Experiment codebase

- To run the Python source code, install the most recent version of Python from python.org.

- During installation under Windows, make sure to check the box to update the PATH.

- For a Windows install, double click the _windows_setup_ batch file to install the needed Python libraries.

- If you are installing an update of this software, you can double click on the _window_update_ batch file to make sure you are using the most recent Python libraries. 

- It is important that the PIP package manager itself is up to date for some of the libraries to install correctly.

In order to use the plate solving function, you also need a database of stars in addition to the program files.
However, plate solving is an optional function. A selection of prepared databases can be found in the **Releases** section.

Plate solving and centroid finding is done using the Tetra3 libary.

For information about Tetra3, see https://tetra3.readthedocs.io/en/latest/api_docs.html.

A simpler way to run this software is to use the Windows executable file, which can be found under **Releases**.
In this case, no Python environment is required. Simply double-click on the executable file or call it using a command line entry.


### **Usage**

For stacking, run the Python file _MEE2024Stacker_ or the Windows executable file.
Select the images to be stacked (the "Light frames").
Dark frame(s) and Flat frame(s) can also be selected, if desired.
Choose the plate solving database (optional).
Select "Show graphics" if you want to view the intermediate graphical analysis (optional).
Choose the number of bright stars to be identified in the stacked image; around 100 seems a reasonable choice.

The plate solver (if used) will identify around 8 to 10 of the brightest stars.
Depending on the database used (Hipparcos or Tycho), you can use a search tool to find the details about the identified stars.

See, for example, https://hipparcos-tools.cosmos.esa.int/HIPcatalogueSearch.html.

HIP identifiers are in the form _aaaaa_.

TYC identifiers are in the form _aaaa-bbbbb-1_.

The file _generate_database_ can be used to create new plate solving databases (with different fields of view and limiting magnitudes).
Both the HIP and TYC star catalogues can be used. A large database can take many hours to generate.

The program will generate a file _MEE_config.txt_ which contains the last used parameters (e.g. directories used).
If this file does not initially exist, it will be created (an error message will be displayed in the console noting that the file does not yet exist).

If you want to save the stacked Darks and Flats for future use, edit the _MEE_config_ file and change "save_dark_flat" from _false_ (the default) to _true_.


