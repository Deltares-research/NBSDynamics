Python-script describtions:
- Bathymetry.py:
	Script to write the bathymetry for the simulations.
- CoralModelD3D_vX.py:
	Script to simulate the development of corals with Delft3D Flexible Mesh.
	(i.e. the biophysical model framework including the coupling with Delft3D-FM.)
	Versions:
	v1 :	Version used in master thesis.
		First version.
		"Full dislodgement".
	v2 :	Work in progress.
		Updated version improving the readability of the code.
		Various model reductions included (as assessed in "CoralModel_ModelReduction.py").
		"Partial dislodgement" instead of "full dislodgement".
- CoralModel_vX.py:
	Script to simulate the development of corals.
	(No coupling with an hydrodynamic model included.)
	Versions:
	v2 :	Based on "CoralModelD3D_v2.py".
		Coupling with Delft3D-FM excluded.
		Improved readability.
		"Partial dislodgement" instead of "full dislodgement".
- CoralModel_ModelReduction.py:
	Modified version of "CoralModelD3D_v1.py" suitable to assess the model reduction.
- CoralModel_PartialRemoval.py:
	Modified version of "CoralModelD3D_v1.py" suitable to assess the partial dislodgement.
- RunTimeD3D.py
	Script to simulate the effect of representative simulation times.
- TimeSeries.py
	Script to write the environmental time-series suitable to run the CoralModel; all versions.

================================================================================================

Steps to get correct Python installation:

- open Anaconda prompt (Anaconda version 3.6.5)

- create environment:
	'conda create --name py36_veg --clone base'

- switch to new environment:
	'conda activate py36_veg'

- downgrade python version (if necessary)
	'conda install python=3.6.5'
	
- install numpy version 1.14.3
	'conda install numpy==1.14.3'

- install scipy version 1.1.0
	'conda install scipy==1.1.0'
	
- install netcdf4 version 1.4.2
	'conda install netcdf4==1.4.2'

- [OPTIONAL] install matplotlib version 2.2.2
	'conda install matplotlib==2.2.2'

- install bmi-python package (local version!)
	'conda install -e c:\software\python\bmi-python-master_20190621\'
	
- install spyder
	'conda install spyder'
	
- start python in spyder interface in the correct environment:
	'spyder'


Or, copy-paste the folder 'py36_veg' in [environment] to the environment-folder of Anaconda:
>	[C:\Users\<DeltaresID>\AppData\Local\Continuum\anaconda3\envs]

(DeltaresID is your Deltares login-account.)