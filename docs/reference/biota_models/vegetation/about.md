# Vegetation model
A biophysical model framework written in Python to make simulations 
on salt marsh species development. The species included are _Spartina anglica_, _Salicornia procumbens_ and _Puccinellia maritima_.

The vegetation bio processes are based on a model developed at Utrecht University, written in Matlab and coupled with Delft3D. 
More information can be found at [Bruckner et al. 2019](https://doi.org/10.1029/2019JF005092) 
and [Bij de Vaate et al. 2020]( https://doi.org/10.1029/2019JF005092).
The translation of the code from Matlab to Python was done by Utrecht University in cooperation with Deltares.

For the hydrodynamics, the model can be coupled to Delft3D Flexible Mesh; 
a hydrodynamic model developed at Deltares 
([more information](https://oss.deltares.nl/web/delft3dfm)). 
To enable this online coupling, certain configurations of Python are required 
([more details](#python-code)).
