# How to run a Vegetation simulation

To fully understand how to create a simulation object in NBSDynamics we recommend first going through the basic guideline [How to create and run a simulation](../../guides/run_simulation).

An example of how a simulation is run can be found in the acceptance tests.

## 1. Simulation structure.
The [Vegetation Delft3d Simulation](../../reference/biota_models/vegetation/vegetation_simulation/) is a concrete implementation of the  [BaseSimulation](../../reference/core/simulation/simulation/#base-simulation) which implements the `SimulationProtocol` already described in the previous mentioned [guide](../../guides/run_simulation#1.-Simulation-structure). As such, we are allowed to define our own interpretations of the simulation. Here is how we build a Vegetation simulation:

* Constants -> [VegetationConstants](../../reference/biota_models/vegetation/vegetation_model/#vegetation-constants). Different species are defined through different constant values, as well as time related constants (e.g. start time and ecological time steps).
    * To initialize the object at least a valid species name is required.
    * This class will load its values looking for the default [vegetation constants file](https://github.com/Deltares/NBSDynamics/blob/master/src/biota_models/vegetation/model/veg_constants.json).
    * We also allow the user to load their own file by doing the following:
    ```python
      VegetationConstants(species='Salicornia', input_file='your_json_filepath_here.json')
    ```
    * Further species can be added by defining their specific parameters using [constants_json_create](https://github.com/Deltares/NBSDynamics/blob/master/src/biota_models/vegetation/model/constants_json_create.py).
* Biota -> [Vegetation](../../reference/biota_models/vegetation/vegetation_model/#vegetation-model)
    * This biota also includes extra characteristics such as [LifeStages](../../reference/biota_models/vegetation/vegetation_model/#src.biota_models.vegetation.model.veg_lifestages.LifeStages).
* Environment -> None (we do not need it for this simulation).
* Hydrodynamics -> Either of the following two:
    * [Delft3D - FlowFMModel](../../reference/core/hydrodynamics/hydromodels/#src.core.hydrodynamics.delft3d.FlowFmModel).
    * [Delft3D - DimrModel](../../reference/core/hydrodynamics/hydromodels/#src.core.hydrodynamics.delft3d.DimrModel).
* Output -> [VegetationOutputWrapper](../../reference/biota_models/vegetation/vegetation_output/#src.biota_models.vegetation.output.veg_output_wrapper.VegOutputWrapper).
    * Our custom wrapper ensures the required output variables are stored to be later evaluated.

## 2. Simulation steps.
When the previous paramaters have been correctly assigned to the object, the simulation is started by using the required methods from [BaseSimulation](../../reference/core/simulation/simulation/#base-simulation) and defined in [VegFlowFmSimulation](reference/biota_models/vegetation/vegetation_simulation/#src.biota_models.vegetation.simulation.veg_delft3d_simulation.VegFlowFmSimulation):

1. Initiate:
    * Configures hydrodynamics and output
    * Validates simulation directories 
    * Initiated vegetation characteristics for all life stages in the class `LifeStages`
    * initializes the output `VegetationOutputWrapper`.

2. Run:
    * When calling the run method ```(sim_run.run())```, the duration of the simulation needs to be specified (in years) (e.g. ```sim_run.run(5)```)
    * If duration is not given, the duration specified in the `VegetationConstants` class will be used.
    * The start date is set to the date specified in `VegetationConstants`.
      * end date = start date + duration.
    * A loop is started over the duration of the simulation (years).
      * Another loop iterates within the previous over the number of ecological time steps per year (coupling times per year) specified in `VegetationConstants`.
      * To get the hydro and morphological variables from Delft-FM, the hydro-morphodynamics are retrieved every day.
      * The coupling and retrieving of the values is specified in the class [Delft3D](.../../reference/core/hydrodynamics/hydromodels/#delft3d)
    * Aggregated values are then created in the class [Hydro_Morphodynamics](../../reference/bio_process/vegetation_processes/#src.biota_models.vegetation.bio_process.veg_hydro_morphodynamics) and retrieved via the method 'get_hydromorph_values'
    * The vegetation dynamics are initiated: 
      * [Mortality and Growth](../../reference/bio_process/vegetation_processes/#src.biota_models.vegetation.bio_process.veg_mortality)
        * criteria for mortality and mortality fractions are determined based on the species and the morpho- &  hydrodynamics 
        * vegetation growth is initiated based on the number of growth days within the current ecological time step
      * [Colonisation](../../reference/bio_process/vegetation_processes/#src.biota_models.vegetation.bio_process.veg_colonisation)
        * method is only called when colonisation is possible during the specific period of the ecological time step 
        * criteria for colonisation are determined based on the species and the morpho- &  hydrodynamics
        * the vegetation characteristics of the initial lifestage are updated based on the possible colonisation
      * [Update_Lifestages](../../reference/vegetation/vegetation_model/#src.biota_models.vegetation.model.veg_model)
        * the life stages are updates (initial to juvenile and juvenile to mature)
        * initial to juvenile always occurs when new vegetation colonized 
        * juvenile to mature only occurs when vegetation in the juvenile stage reached the maximum years in that life stage
        * if the maximum age of vegetation is reached, the vegetation is removed
    * The results are exported using the methods defined in [VegetationMapOutput](../../reference/output/vegetation_output/#src.biota_models.vegetation.output.veg_output_model) and [VegetationHisOutput](../../reference/output/vegetation_output/#src.biota_models.vegetation.output.veg_output_model)

3. Finalize (Finalize simulation).

### Vegetation Delft3D Simulation Example
A simulation using the [VegFlowFmSimulation](../../reference/biota_models/vegetation/vegetation_simulation/#Vegetation-Delft3D) object. This example is in our test bench, although we only run it locally due to the DIMR dependencies not being available at the repo level, and makes use of a Pydantic approach to simplify how to initialize an object.

```python
from src.biota_models.vegetation.model.veg_constants import VegetationConstants
from src.biota_models.vegetation.model.veg_model import Vegetation
from src.biota_models.vegetation.simulation.veg_delft3d_simulation import (
    VegFlowFmSimulation,
)
# 1. Define attributes
test_dir = TestUtils.get_local_test_data_dir("sm_testcase6")
# We need to specify where the DIMR directory (WITH ALL THE SHARED BINARIES) is located.
dll_repo = TestUtils.get_external_repo("DimrDllDependencies")
kernels_dir = dll_repo / "kernels"
output_dir = test_dir / "output"
test_case = test_dir / "input" / "MinFiles"

# 2. Prepare model.
# Create vegetation constants based on the above species.
veg_constants = VegetationConstants(species="Salicornia")
sim_run = VegFlowFmSimulation(
    working_dir=test_dir,
    constants=veg_constants,
    hydrodynamics=dict(
        working_dir=test_dir / "d3d_work",
        d3d_home=kernels_dir,
        dll_path=kernels_dir / "dflowfm_with_shared" / "bin" / "dflowfm.dll",
        definition_file=test_case / "fm" / "test_case6.mdu",
    ),
    output=dict(
        output_dir=output_dir,
        map_output=dict(output_params=dict()),
        his_output=dict(
            output_params=dict(),
        ),
    ),
    biota=Vegetation(species="Salicornia", constants=veg_constants),
)

# 3. Run simulation.
sim_run.initiate()
sim_run.run()
sim_run.finalise()
```