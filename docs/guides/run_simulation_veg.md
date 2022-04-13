# How to run a Vegetation simulation in NBSDynamics

To fully understand how to create a simulation object in NBSDynamics we recommend first going through the basic guideline [How to create and run a simulation](../../guides/run_simulation).

An example of how a simulation is run can be found in the [acceptance tests](../../test/test_acceptance.py#TestAcceptance::test_given_veg_case_runs)

## Simulation structure.

The simulation is started using the [simtest_saltmars_test](../../test/test_data/simtest_saltmarsh_test.py) file. 
Here the different paths need to be specified. 
An object of the class [VegFlowFmSimulation](../../reference/simulation/vegetation_simulation/#src.biota_models.vegetation.simulation.veg_delft3d_simulation) (which inherits from the BaseSimulation) is created. 
Within the object, the vegetation species (in constants and vegetation) are required as an input. 
Different species are defined through different constants. 
In [Constants](../../reference/common/common/#src.biota_models.vegetation.model.veg_constants) a json file is called which contains the species specific vegetation parameters. 
Additional time related constants (e.g. start time and ecological time steps) are defined in [Constants](../../src/core/common/common/#src.biota_models.vegetation.model.veg_constants) and can be changed manually there.  
Further species can be added by defining their specific parameters using [constants_json_create](../../src/core/common/common/#src.biota_models.vegetation.model.constants_json_create).

With those parameters assigned to the object, the simulation is started by using the following methods (specified in the [BaseSimulation](../../reference/simulation/vegetation_simulation/#src.biota_models.vegetation.simulation.veg_base_simulation)):

* initiate
  * configures hydrodynamics and output
  * validates simulation directories 
  * initiated vegetation characteristics for all life stages in the class [LifeStages](../../reference/vegetation/vegetation_model/#src.biota_models.vegetation.model.veg_lifestages)
  * initializes the output ([VegOutputWrapper](../../reference/output/vegetation_output/#src.biota_models.vegetation.output.veg_output_wrapper))

* run
  * when calling the run method (sim_run.run()), the duration of the simulation needs to be specified (in years) (e.g. sim_run.run(5))
  * if duration is not given, the duration specified in the [Constants](../../reference/common/common/#src.biota_models.vegetation.model.veg_constants) class will be used 
  * the start date is set to the date specified in [Constants](../../reference/common/common/#src.biota_models.vegetation.model.veg_constants)
  * end date = start date + duration 
  * a loop is started over the duration of the simulation (years)
  * inside of that loop another loop iterates over the number of ecological time steps per year (coupling times per year) specified in [Constants](../../reference/common/common/#src.biota_models.vegetation.model.veg_constants)
  * to get the hydro and morphological variables from Delft-FM, the hydro-morphodynamics are retrieved every day.
    * the coupling and retrieving of the values is specified in the class [Delft3D](.../../reference/hydrodynamics/hydromodels/#delft3d)
  * aggregated values are then created in the class [Hydro_Morphodynamics](../../reference/bio_process/vegetation_processes/#src.biota_models.vegetation.bio_process.veg_hydro_morphodynamics) and retrieved via the method 'get_hydromorph_values'
  * the vegetation dynamics are initiated: 
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
  * The results are exported using the methods defined in [MapOutput](../../reference/output/vegetation_output/#src.biota_models.vegetation.output.veg_output_model) and [HisOutput](../../reference/output/vegetation_output/#src.biota_models.vegetation.output.veg_output_model)

* finalize (Finalize simulation)

### Vegetation Delft3D Simulation
```python
# 1. Define attributes
test_dir = TestUtils.get_local_test_data_dir("sm_testcase6")
# We need to specify where the DIMR directory (WITH ALL THE SHARED BINARIES) is located.
dll_repo = TestUtils.get_external_repo("DimrDllDependencies")
kernels_dir = dll_repo / "kernels"
output_dir = test_dir / "output"
test_case = test_dir / "input" / "MinFiles"

# 2. Prepare model.
sim_run = VegFlowFmSimulation(
    working_dir=test_dir,
    constants=VegetationConstants(species="Salicornia"),
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
    biota=Vegetation(species="Salicornia"),
)

# 3. Run simulation.
sim_run.initiate()
sim_run.run()
sim_run.finalise()
```