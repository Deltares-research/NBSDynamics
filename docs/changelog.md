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
