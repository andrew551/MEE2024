# MEE2024
Modern Eddington Experiment codebase

- Install the most recent version of Python from python.org. During Windows installation, check the box to update the PATH.

- For Windows, double click the _windows_setup_ batch file to install the needed Python libraries.
If you are installing an update of this software, double click on _window_update_ batch file to make sure you are using the most recent Python libraries. 
It is important that the PIP package manager itself is up to date for some of the libraries to install correctly.

In order to use the plate solving function, you must download a database of stars for in addition to these program files.
However, plate solving is an optional function. A selection of prepared databases can be found in the release section.

Plate solving and centroid finding is done using the Tetra3 libary.
For information about Tetra3, see https://tetra3.readthedocs.io/en/latest/api_docs.html.


### **Usage**

For stacking, run the Python file _MEE2024Stacker_. Select the images to be stacked (the "Light frames").
Dark frame(s) and Flat frame(s) can also be selected, if desired.
Choose the plate solving database (optional).
Select "Show graphics" if you want to view the intermediate graphical analysis (optional).
Choose the number of bright stars to be identified in the stacked image. Around 50 seems a reasonable choice.
The plate solver (if used) will identify around 8 to 10 of the brightest stars.

Depending on the database used (Hipparcos or Tycho), you can use a search tool to find the details about the identified stars.

See, for example, https://hipparcos-tools.cosmos.esa.int/HIPcatalogueSearch.html.
TYC identifiers are in the form _aaaa-bbbbb-1_.

HIP identifiers are in the form _ccccc_.


