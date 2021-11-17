## v0.6.0 (2021-11-17)

### Feat

- **src/core/simulation/base_simulation.py**: Added `vanilla` simulation, adapted reef models to be pydantic
- **src/core/hydrodynamics/transect.py**: Transect initializes now as a pydantic model. Smaller related refactorings
- **src/core/hydrodynamics/delft3d.py**: dll_path can be now manually set by the user
- **src/core/hydrodynamics/delft3d.py**: Now the user can input their own path to the dll to be run

### Fix

- **src/core/simulation/coral_delft3d_simulation.py**: Minor corrections to the outpoint definition
- **src/core/hydrodynamics/delft3d.py**: Minor fix to xy_coordinates inital values
- **src/core/output/output_wrapper.py**: Adapted output_wrapper to ensure an output_dir is given for all the output_model instances
- **src/core/hydrodynamics/delft3d.py**: Renamed method, added explanation of what needs to be done

### Refactor

- **test/core/simulation**: Removed unnecessary class method, adapted tests, fixed failing tests
- **src/core/simulation/base_simulation.py**: Adapted simulation logic to initate hydrodynamics in a pydantic way

## v0.5.1 (2021-11-16)

### Fix

- **bio_process/test_morphology.py**: removed bug from test fixture

## v0.5.0 (2021-11-15)

### Feat

- coral model as protocol (#66)
- simulation as protocol (#65)

## v0.4.0 (2021-11-15)

### Fix

- **test/test_acceptance.py**: corrected numpy usage
- **src/core/output/output_wrapper.py**: Output wrapper now creates the output dir if it was not there already

### Feat

- **src/core/simulation.py;test/test_acceptance.py**: Now simulations can delegate the initial setup of a hydrodynamic model, adapted test for d3d case
- **src/core/output/output_model.py;src/core/output/output_wrapper.py;src/core/hydrodynamics/factory.py;src/core/simulation.py**: Refactor output wrapper and simulation classes so they can be instantiated with less attributes

## v0.3.0 (2021-11-12)

### Feat

- Create test bmi coral (#63)

## v0.2.0 (2021-11-11)

### Fix

- **src/core/environment.py;src/core/simulation.py;test/core/test_environment.py**: Extended validation for dates so they can be added as string from the initialization
- **src/core/output/output_wrapper.py**: Now we add the dict attributes from the wrapper to the output models
- **src/core/output/output_model.py;src/core/output/output_wrapper.py;src/core/simulation.py;test/core/output/test_wrapper.py**: Now we correctly initialize the map_output model. Adapted test output wrapper
- **src;test**: Output model generates again the netcdf files correctly.
- **src/tools/plot_output.py**: Corrected call to plot tool

### Feat

- **src/core/environment.py;src/core/simulation.py**: Adapted environment and related classes to pydantic approach

- **src/core/coral_model.py;src/core/simulation.py;test/test_acceptance.py**: Coral integrated in simulation

### Refactor

- **src/core/environment.py;src/core/simulation.py;test/core/test_simulation**: Improved environment as class

- **src/core/environment.py;src/core/output_model.py;src/core/simulation.py;test/core/test_output**: Applying pydantic to Output model.

- **src/core/simulation**: Added validator to simulation constants property

- **src/core/simulation.py;test/test_acceptance**: Adapted code for better pydantic usage

- **src/core/output_model**: Fixed failing tests for TestOutput

- **src/core/output_model**: Minor corrections to the model

- **src/core/output_model**: Small fix, however model still not running.

- **src/core/output_model**: fixed ini/update his/map

- **src/core/environment**: Added extra logic to accept str as path.

- **src/core/output/output_model**: Extracted two submodels from Output, moved into its own module.

- **src/core/output/output_protocol.py;src/core/output/output_wrapper**: Separating concepts to avoid code duplication

- **src/core/output/output_model.py;src/core/output/output_protocol.py;src/core/output/output_wrapper**: Refactor output module for better maintainability and reducing its complexity.

- **test/core/output/test_wrapper**: Moved test to mirror src structure

- **src/core/output/**: Extended docstrings for output_protocol; Generated coverage tests for output_model. Fixed simulation calls to output initialization.

- **test_output_wrapper**: Renamed filename to match src tested file


## v0.1.4 (2021-11-11)

### Fix

- quality gate fix (#62)

### Feat

- Create model input

## v0.1.3 (2021-11-08)

### Refactor

- **src/core/coral_only.py**: Extracted coral only for better maintainability.
- **src/core/output_model.py;src/core/utils.py**: further type hinting.
- **src/core/output_model.py**: Fixed setting of xy_stations.
- **core/utils.py**: Added more type hinting.
- **core/utils.py**: Adding type hints.
- **core/loop.py;core/output_model.py;core/utils.py**: Extracted output model logic into its own class. Introduced new libraries.

## v0.1.2 (2021-11-05)

## v0.1.1 (2021-11-05)

### Fix

- **.github/workflows/ci.yml**: improve code coverage (#44)

## v0.1.0 (2021-11-04)

### Fix

- Fix merged conflict

### Feat

- Pull-request 31 normalize versioning (#42)

## v0.0.4 (2021-11-04)

### Fix

- **pyproject.toml**: Corrected version file for core directory.
- **pyproject.toml**: Changed bump pattern and map to include refactor and docs.

## v0.0.3 (2021-11-04)

### Refactor

- **hydrodynamics.py**: Refactor the update method to use update_ (#30)

## v0.0.2 (2021-10-29)

### Fix

- **environment.py-utils.py**: Fixed bugs described in sonarcloud (#26)
- **Removed-unused-reference.**: removed unused reference
