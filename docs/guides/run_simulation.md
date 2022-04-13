# How to run a simulation in NBSDynamics

The current package offers the user the possibility to run a simulation with the built-in simulation types, or to create their own simulation with their custom attributes.

## 1. Simulation structure.
A simulation is based on the [SimulationProtocol](../../reference/core/simulation/simulation/#simulation-protocol). Thus its required attributes are as follows:

* [Constants](../../reference/core/common/common/#constants). Definition of constants to be used during the simulation.
* [Coral](../../reference/coral/coral/#coral-model). Required to represent a morphological model of a coral.
* [Environment](../../reference/core/common/common/#environment).
* [Hydrodynamics](../../reference/core/hydrodynamics/hydromodels/). Defines the type of simulation that will be run. As per version v0.8.0 the following types are available:
    * [Reef0D](../../reference/core/hydrodynamics/hydromodels/#reef-0d)
    * [Reef1D](../../reference/core/hydrodynamics/hydromodels/#reef-1d)
    * [Transect](../../reference/core/hydrodynamics/hydromodels/#transect)
    * [Delft3D](../../reference/core/hydrodynamics/hydromodels/#delft3d). Simulation through usage of a BMI runner:
        * [Delft3D - FlowFMModel](../../reference/core/hydrodynamics/hydromodels/#src.core.hydrodynamics.delft3d.FlowFmModel). Currently under work.
        * [Delft3D - DimrModel](../../reference/core/hydrodynamics/hydromodels/#src.core.hydrodynamics.delft3d.DimrModel). Currently under work.
* [Output](../../reference/core/output/output/#wrapper). Required to define what output should be stored, how, where and when.

## 2. 'Vanilla' Simulation.
The user has the possibility to create its custom simulation by calling to the most simple simulation class [`Simulation`](../../reference/core/simulation/simulation/#src.core.simulation.base_simulation.Simulation). This class allows the user to combine their own set of `Constants`, `Coral`, `Environment`, `Hydrodynamics` and `Output` and then call the predefined `initiate`, `run` or `finalise` methods.

Keep in mind this way of running a simulation implies the user will have to manually configure the Simulation attributes.

## 3. Built-in Simulations.
A built-in simulation allows the user to only worry about given the required input parameters and then letting the object pre-configure all the data as required for a regular run.

### Available
As per version NBSDynamics v.0.8.0 the following simulations are fully validated:
#### Coral Transect Simulation
A simulation using the [CoralTransectSimulation](../../reference/simulation/simulation/#coral-transect) object. Which will build the simulation around a `Transect` hydrodynamic model. We provide here an example (currently used for testing) of its usage with a Pydantic approach:
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
    coral=dict(
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

### Work in progress
The following simulations are defined, however their status is not yet final and can therefore not be guaranteed to work.

#### Coral FlowFm Simulation
Open issue: [#68 Fix Delft3D - FlowFMModel run.](https://github.com/Deltares/NBSDynamics/issues/68)

A simulation using the [CoralFlowFmSimulation](reference/simulation/simulation/#src.biota_models.coral.simulation.coral_delft3d_simulation.CoralFlowFmSimulation) object. Which will build the simulation around a `FlowFMModel` hydrodynamic model.

#### Coral Dimr Simulation
Open issue: [#69 Fix / Implement Delft3D - DIMR run.](https://github.com/Deltares/NBSDynamics/issues/69)

A simulation using the [CoralDimrSimulation](../../reference/simulation/simulation/#src.biota_models.coral.simulation.coral_delft3d_simulation.CoralDimrSimulation) object. Which will build the simulation around a `DimrModel` hydrodynamic model.
