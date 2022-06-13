[![ci](https://github.com/Deltares/NBSDynamics/actions/workflows/ci.yml/badge.svg)](https://github.com/Deltares/NBSDynamics/actions/workflows/ci.yml)
[![docs](https://github.com/Deltares/NBSDynamics/actions/workflows/docs.yml/badge.svg)](https://github.com/Deltares/NBSDynamics/actions/workflows/docs.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Deltares_NBSDynamics&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=Deltares_NBSDynamics)
![Sonar Coverage](https://img.shields.io/sonar/coverage/Deltares_NBSDynamics?logo=SonarCloud&server=https%3A%2F%2Fsonarcloud.io&?style=plastic&logo=appveyor)

![GitHub release (latest by date)](https://img.shields.io/github/v/release/Deltares/NBSDynamics)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/Deltares/NBSDynamics)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Official Documentation.
Check our official GitHub pages documentation at [https://deltares.github.io/NBSDynamics/](https://deltares.github.io/NBSDynamics/).

# Quickguide
Thanks to the latest version of Pypi you can get all setup by just installing the package in the edit mode, so we offer the following options:

* Contributing to the project:
    * [Full guide](https://deltares.github.io/NBSDynamics/guides/contribute/)
    * Quick installation (without poetry) __for development__: 
        * Navigate to the checked-out directory.
        * Update your 'pip' to the latest version.
        * Install the package in edit mode:
        ```cli
        pip install -e .
        ```
* Using the package as an external library:
    ```cli
    pip install git+https://github.com/Deltares/NBSDynamics.git
    ```
    | We recommend installing the latest available release at the time instead of from 'master'. For that just add @branch_name at the end of the previous pip call. 

### Potential Errors
We found out that packages such as `netcdf4` or `pypiwin32` / `pywin32` might give problems. In case you run against said problems while installing the package we recommend installing them beforehand.

# Current supported models.
Currently we support the Vegetation and Coral models. More details of how to run them can be found at their respective documentation pages.
In addition, quick links on how to run models in this package:
* [Basics: How to run 'any' model](https://deltares.github.io/NBSDynamics/guides/run_simulation/)
* [How to run a Vegetation Model](https://deltares.github.io/NBSDynamics/guides/run_simulation_veg/)
* [How to run a Coral Model](https://deltares.github.io/NBSDynamics/guides/run_simulation_coral/)

