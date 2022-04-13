# How to run a Coral Simulation

We have defined an intermediate (internal) [CoralSimulation](../../reference/biota_models/coral/coral_simulation/#base-coral-simulation) object. This one allows us to easily implement other simulation configurations by simply changing its hydrodynamic properties.

Currently we offer definitions for Coral Transect and Coral Delft3D configurations.

### Coral Transect Simulation
A simulation using the [CoralTransectSimulation](../../reference/biota_models/coral/coral_simulation/#coral-transect) object. Which will build the simulation around a `Transect` hydrodynamic model. We provide here an example (currently used for testing) of its usage with a Pydantic approach:
```python
# 1. Define attributes.
test_dir = Path("transect_run")
input_dir = test_dir / "input"
output_dir = test_dir / "output"

# 2. Prepare model.
# Define the basic Simulation object, indicating already here the type of hydrodynamics
run_trans = CoralTransectSimulation(
    working_dir=test_dir,
    constants=input_dir / "coral_input.txt",
    environment=dict(
        light=input_dir / "TS_PAR.txt",
        temperature=input_dir / "TS_SST.txt",
        storm=input_dir / "TS_stormcat2.txt",
        dates=("2000-01-01", "2100-01-01"),
    ),
    hydrodynamics=dict(
        definition_file=input_dir / "TS_waves.txt",
        config_file=input_dir / "config.csv",
    ),
    output=dict(
        output_dir=test_dir / "output",
        map_output=dict(output_params=dict(fme=False)),
        his_output=dict(output_params=dict(fme=False)),
    ),
    biota=dict(
        dc=0.125,
        hc=0.125,
        bc=0.1,
        tc=0.1,
        ac=0.2,
        Csp=0.6,
    ),
)

# 3. Run simulation
run_trans.initiate()
run_trans.run()
run_trans.finalise()
```

## Work in progress
The following simulations are defined, however their status is not yet final and can therefore not be guaranteed to work.

### Coral FlowFm Simulation
Open issue: [#68 Fix Delft3D - FlowFMModel run.](https://github.com/Deltares/NBSDynamics/issues/68)

A simulation using the [CoralFlowFmSimulation](reference/simulation/simulation/#src.biota_models.coral.simulation.coral_delft3d_simulation.CoralFlowFmSimulation) object. Which will build the simulation around a `FlowFMModel` hydrodynamic model.

### Coral Dimr Simulation
Open issue: [#69 Fix / Implement Delft3D - DIMR run.](https://github.com/Deltares/NBSDynamics/issues/69)

A simulation using the [CoralDimrSimulation](../../reference/simulation/simulation/#src.biota_models.coral.simulation.coral_delft3d_simulation.CoralDimrSimulation) object. Which will build the simulation around a `DimrModel` hydrodynamic model.
