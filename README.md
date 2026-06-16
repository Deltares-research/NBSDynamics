[![ci](https://github.com/Deltares/NBSDynamics/actions/workflows/ci.yml/badge.svg)](https://github.com/Deltares/NBSDynamics/actions/workflows/ci.yml)
[![docs](https://github.com/Deltares/NBSDynamics/actions/workflows/docs.yml/badge.svg)](https://github.com/Deltares/NBSDynamics/actions/workflows/docs.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Deltares_NBSDynamics&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=Deltares_NBSDynamics)
![Sonar Coverage](https://img.shields.io/sonar/coverage/Deltares_NBSDynamics?logo=SonarCloud&server=https%3A%2F%2Fsonarcloud.io&?style=plastic&logo=appveyor)

![GitHub release (latest by date)](https://img.shields.io/github/v/release/Deltares/NBSDynamics)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/Deltares/NBSDynamics)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Official Documentation.
The present NBSDynamics package is the result of a long development history, with direct and indirect contributions by multiple individuals and organisations. Key contributions are highlighted here, please reach out if you consider this incomplete or incorrect or if you want to contribute yourself.  
## Origins
The principle of simulating biogeomorphology on an engineering scale, using Delft3D and a relatively simple ecological model in Matlab, was first described in a paper by Stijn Temmerman and others from NIOZ (then NIOO-CEME; Temmerman et al., 2007) for a single-species salt marsh case, enabled by the incorporation of vegetation-specific drag formulations by Martin Baptist and colleagues (Deltares & TUDelft; Baptist et al., 2007). This toolset was further developed for riverine applications, with multiple terrestrial vegetation types and life stages by Mijke van Oorschot (Utrecht University & Deltares, e.g. Van Oorschot et al., 2015). Subsequent work by Muriel Brückner and colleagues (Utrecht University; Brückner et al., 2019 and 2020; Bij de Vaate et al., 2020) involved a return to the coastal/estuarine environment, modelling juvenile and mature salt marsh growth including several marsh species as well as microphytobenthos and bioturbation. Mangroves were added by Danghan Xie and colleagues (Utrecht University; Xie et al., 2020). Meanwhile, Üwe Best (IHE Delft, Best et al., 2018) added wave action via a SWAN model to study single-species marsh development under sea level rise, not only to simulate wave forces determining survival of the vegetation but also to simulate the role of waves in resuspending sediment from the mudflat. 
## New technology: BMI, Python, Flexible Mesh
Although the Delft3D4-Matlab combination offered a lot of freedom to incorporate ecological processes via Matlab code without having to interfere with the computational core in Delft3D, the fundamental shortcoming was that these model components only interacted via in-and output files requiring frequent restarts, which was error-prone and slow. Moreover, not every potential user could afford a Matlab license. To alleviate this, Deltares developed a new toolset based on a combination of Delft3D - Flexible Mesh and Python, using BMI (Basic Model Interface; Peckham et al., 2013 and Hutton et al., 2020) by CSDMS of University of Colorado) for model running and information exchange via memory. Pim Willemsen and colleagues (Deltares, NIOZ and Twente University) demonstrated this combination for salt marsh development in relation to wave action, also using D-Waves, with a simple single-species population model similar to that of Temmerman et al. (2007) in Willemsen et al. (2022). The same combination was used by Gijs Hendrickx (TUDelft, Deltares, USGS; Hendrickx et al., 2021) to simulate reef-flow-wave interactions in coral reef development. 
## Present NBSDynamics
With so many applications by different users, it became apparent that the Python code was indeed flexible enough to implement environment-specific processes but it did not lend itself for easy re-use and incorporation of contributions by other users as it lacked a systematic, testable setup. To address this, Deltares and Utrecht university agreed to restructure and refactor the Python code to establish a solid, maintainable base version. This was done based on the Python code structure by Gijs Hendrickx, in combination with the salt marsh code contents by Muriel Brückner (Brückner et al., 2019; 2020; Bij de Vaate et al., 2020). The coding for the resulting NBSDynamics package was largely done by Carles Sobriano Pérez from Deltares, with testing and description based on the Bij de Vaate case (bij de Vaate et al., 2020) by Debora de Toledo Alvez and Sarah Dzimballa from Utrecht University.

### References of directly contributing authors: 
Best, Ü. S. N., van der Wegen, M., Dijkstra, J., Willemsen, P. W. J. M., Borsje, B. W., & Roelvink, D. (2018). Do salt marshes survive sea level rise? Modelling wave action, morphodynamics and vegetation dynamics. Environmental Modelling & Software, 109, 152–166. https://doi.org/10.1016/j.envsoft.2018.08.004

Brückner, M. Z. M., Schwarz, C., & Dijk, W. M. van. (2019). Salt Marsh Establishment and Eco-Engineering Effects in Dynamic Estuaries Determined by Species Growth and Mortality. JGR Earth Surface, 124(12), 2962–2986. https://doi.org/10.1029/2019JF005092

Brückner, M. Z. M., Braat, L., Schwarz, C., & Kleinhans, M. G. (2020). What Came First, Mud or Biostabilizers? Elucidating Interacting Effects in a Coupled Model of Mud, Saltmarsh, Microphytobenthos, and Estuarine Morphology. Water Resources Research, 56(9). https://doi.org/10.1029/2019WR026945

Hendrickx, G. G., Herman, P. M. J., Dijkstra, J. T., Storlazzi, C. D., & Toth, L. T. (2021). Online-coupling of widely-ranged timescales to model coral reef development. Environmental Modelling & Software, 143, 105103. https://doi.org/10.1016/j.envsoft.2021.105103

Hutton, E.W.H., Piper, M.D., and Tucker, G.E., 2020. The Basic Model Interface 2.0: A standard interface for coupling numerical models in the geosciences. Journal of Open Source Software, 5(51), 2317, https://doi.org/10.21105/joss.02317.

Peckham, S.D., Hutton, E.W., and Norris, B., 2013. A component-based approach to integrated modeling in the geosciences: The design of CSDMS. Computers & Geosciences, 53, pp.3-12, http://dx.doi.org/10.1016/j.cageo.2012.04.002.

Temmerman, S., Bouma, T. J., van de Koppel, J., van der Wal, D., de Vries, M. B., & Herman, P. M. J. (2007). Vegetation causes channel erosion in a tidal landscape. Geology, 35(7), 631. https://doi.org/10.1130/G23502A.1

Bij de Vaate, I., Brückner, M. Z. M., Kleinhans, M. G., & Schwarz, C. (2020). On the impact of salt marsh pioneer species-assemblages on the emergence of intertidal channel networks. Water Resources Research, 0–2. https://doi.org/10.1029/2019WR025942

van Oorschot, M., Kleinhans, M., Geerling, G., & Middelkoop, H. (2015). Distinct patterns of interaction between vegetation and morphodynamics. Earth Surface Processes and Landforms, 41(6), 791-808. https://doi.org/10.1002/esp.3864

Willemsen, P. W. J. M., Smits, B. P., Borsje, B. W., Herman, P. M. J., Dijkstra, J. T., Bouma, T. J., & Hulscher, S. J. M. H. (2022). Modeling Decadal Salt Marsh Development: Variability of the Salt Marsh Edge Under Influence of Waves and Sediment Availability. Water Resources Research, 58(1), 1–23. https://doi.org/10.1029/2020wr028962

Xie, D., Schwarz, C., Brückner, M. Z. M., Kleinhans, M. G., Urrego, D. H., Zhou, Z., & Van Maanen, B. (2020). Mangrove diversity loss under sea-level rise triggered by bio-morphodynamic feedbacks and anthropogenic pressures. Environmental Research Letters, 15(11), 1-12. https://doi.org/10.1088/1748-9326/abc122

### References of users that helped to improve the software, e.g. by adding functionality, testing, troubleshooting or requiring additional functionality:

Dzimballa, S., Willemsen, P. W. J. M., Kitsikoudis, V., Borsje, B. W., & Augustijn, D. C. M. (2025). Numerical modelling of biogeomorphological processes in salt marsh development: Do short-term vegetation dynamics influence long-term development? Geomorphology, 471, 109534. https://doi.org/10.1016/J.GEOMORPH.2024.109534

Gijsman, R., Horstman, E. M., Swales, A., Balke, T., Willemsen, P. W. J. M., Wal, D. van der, & Wijnberg, K. M. (2024). Biophysical Modeling of Mangrove Seedling Establishment and Survival Across an Elevation Gradient With Forest Zones. Journal of Geophysical Research: Earth Surface, 129(5), e2024JF007664. https://doi.org/10.1029/2024JF007664

Jean Louis, M., Dijkstra, J., Quirk, T., Rovai, A., Willemsen, P., & Hiatt, M. (2026). The role of seasonal vegetation dynamics in shaping river delta channel networks and morphodynamics. Geomorphology, 110327. https://doi.org/10.1016/J.GEOMORPH.2026.110327


Check our official GitHub pages documentation at [https://deltares-research.github.io/NBSDynamics/](https://deltares-research.github.io/NBSDynamics/).

# Quickguide
Thanks to the latest version of Pypi you can get all setup by just installing the package in the edit mode, so we offer the following options:

* Contributing to the project:
    * [Full guide](https://deltares-research.github.io/NBSDynamics/guides/contribute/)
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
* [Basics: How to run 'any' model](https://deltares-research.github.io/NBSDynamics/guides/run_simulation/)
* [How to run a Vegetation Model](https://deltares-research.github.io/NBSDynamics/guides/run_simulation_veg/)
* [How to run a Coral Model](https://deltares-research.github.io/NBSDynamics/guides/run_simulation_coral/)

