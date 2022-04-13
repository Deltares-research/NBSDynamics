[![ci](https://github.com/Deltares/NBSDynamics/actions/workflows/ci.yml/badge.svg)](https://github.com/Deltares/NBSDynamics/actions/workflows/ci.yml)
[![docs](https://github.com/Deltares/NBSDynamics/actions/workflows/docs.yml/badge.svg)](https://github.com/Deltares/NBSDynamics/actions/workflows/docs.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Deltares_NBSDynamics&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=Deltares_NBSDynamics)
![Sonar Coverage](https://img.shields.io/sonar/coverage/Deltares_NBSDynamics?logo=SonarCloud&server=https%3A%2F%2Fsonarcloud.io&?style=plastic&logo=appveyor)

![GitHub release (latest by date)](https://img.shields.io/github/v/release/Deltares/NBSDynamics)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/Deltares/NBSDynamics)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Official Documentation.
Check our official GitHub pages documentation at [https://deltares.github.io/NBSDynamics/](https://deltares.github.io/NBSDynamics/).

# Current supported models.
Currently we support the Vegetation and Coral models. More details of how to run them can be found at their respective documentation pages.
In addition, quick links on how to run models in this package:
* [Basics: How to run 'any' model](https://deltares.github.io/NBSDynamics/guides/run_simulation/)
* [How to run a Vegetation Model](https://deltares.github.io/NBSDynamics/guides/run_simulation_veg/)

## About the Vegetation Model
(TODO)

## About the Coral Model
A biophysical model framework written in Python to make simulations 
on coral development based on four environmental factors: 
(1) light; (2) flow; (3) temperature; and (4) acidity. 
For the hydrodynamics, the model can be coupled to Delft3D Flexible Mesh; 
a hydrodynamic model developed at Deltares 
([more information](https://oss.deltares.nl/web/delft3dfm)). 
To enable this online coupling, certain configurations of Python are required 
([more details](#python-code)).

**Note:** This model is still in its beta and further development is still being done. 
``coral_model_v0`` is used for the study and is rewritten (``coral_model``) to enhance collaboration.
(The original version has not been written efficiently and is hard to follow for outsiders.)
More information on this version control [here](#version-control).

### Biophysics <a name="biophsics"></a>
This biophysical model framework is part of the result of a 
[master thesis](https://repository.tudelft.nl/islandora/object/uuid%3Ae211380e-3f92-4afe-b371-f1e87b0c3bbd?collection=education) 
of which the key findings are published in *Environmental Modelling and Software*
(the paper can be found [here](https://www.sciencedirect.com/science/article/pii/S1364815221001468?via%3Dihub)).

The biophysical relations used in the biophysical model framework are mainly process-based, 
where for the acidity the proxy of the aragonite saturation state is used. 
Furthermore, both photo- and thermal-acclimatisation are included, 
which result in a dynamic behaviour of the corals to their environment. 
Hence, the corals are modelled such that they can adapt to changing environmental conditions over time.

For more details on the biophysics, reference is made to the 
[master thesis](https://repository.tudelft.nl/islandora/object/uuid%3Ae211380e-3f92-4afe-b371-f1e87b0c3bbd?collection=education) 
and the [paper](https://www.sciencedirect.com/science/article/pii/S1364815221001468?via%3Dihub) that substitute this
repository.
