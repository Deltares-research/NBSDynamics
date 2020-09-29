# CoralModel
A biophysical model framework written in Python to make simulations on coral development based on four environmental factors: (1) light; (2) flow; (3) tempreature; and (4) acidity. For the hydrodynamics, the model can be coupled to Delft3D Flexible Mesh; a hydrodynamic model developed at Deltares ([more information](https://oss.deltares.nl/web/delft3dfm)). To enable this online coupling, certain configurations of Python are required (more details further down below).

**Note:** This model is still in its beta and further development is still being done.

## Biophysics
This biophysical model framework is part of the result of a [master thesis](https://repository.tudelft.nl/islandora/object/uuid%3Ae211380e-3f92-4afe-b371-f1e87b0c3bbd?collection=education) of which the key findings are to be published in *Journal (TBD)* (**note:** paper is still in development, 29-09-2020).

The biophysical relations used in the biophysical model framework are mainly process-based, where for the acidity the proxy of the aragonite saturation state is used. Furthermore, both photo- and thermal-acclimatisation are included, which result in a dynamic behaviour of the corals to their environment. Hence, the corals are modelled such that they can adapt to changing environmental conditions over time.

For more details on the biophysics, reference is made to the [master thesis](https://repository.tudelft.nl/islandora/object/uuid%3Ae211380e-3f92-4afe-b371-f1e87b0c3bbd?collection=education) *and the [paper]()* (**note:** paper is still in development, 29-09-2020).

## Python code
The Python code is written in Python 3 and makes use of various packages. Not all of these packages are automatically included in the standard library of Python, such as `NetCDF4` (can be downloaded [here](http://www.ldf.uci.edu/~gohlke/pythonlibs/#netcdf4)). In case the biophysical model framework is to be coupled to Delft3d Flexible Mesh, the `bmi.wrapper` package is also required (can be downloaded [here](https://github.com/openearth/bmi-python)).

The settings of Python and other packages for the online coupling to work properly are the following:
* Python version 3.6.5
* NumPy version 1.14.3
* SciPy version 1.1.0
* NetCDF4 version 1.4.2
* Matplotlib version 2.2.2
* BMI-Python

**Note:** These requirements are only required in case the biophysical model framework is to be coupled with Delft3D Flexible Mesh.
