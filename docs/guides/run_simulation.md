# How to run a simulation in NBSDynamics

The current package offers the user the possibility to run a simulation with the built-in simulation types, or to create their own simulation with their custom attributes.

## 1. Simulation structure.
A simulation is based on the [SimulationProtocol](../../reference/core/simulation/simulation/#simulation-protocol). Thus its required attributes are as follows:

* [Constants](../../reference/core/common/common/#constants). Definition of constants to be used during the simulation.
* [Biota](../../reference/core/biota/#biota-model). Required to represent a biota entity in the simulation.
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
As per version NBSDynamics v.0.8.1 the following simulations are fully validated:

* [Coral Transect Simulation](../../guides/run_simulation_coral/#coral-transect-simulation)
* [Vegetation Delft3D Simulation](../../guides/run_simulation_veg/#coral-transect-simulation)
