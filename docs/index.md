# NBSDynamics documentation

On this pages diverse information can be found related to this project.

## Project layout
The project is currently being structured as follows:

    docs/               # Contains the documentation related to this project.
        guides/         # Contains the documentation related to installation, usage and contribution to this project.
        reference/      # Contains references to the modules. Docstrings will be used to generate automatically documentation.        
    src/
        biota_models/   # Module containing all biota species models that inherit or use the modules from core/
            coral/      # Module containing the coral model definition.
                bioprocess/ # Contains all the different bio processes required by the coral model.
                model/  # Defines the coral model and related (such as constants).
                output/ # Contains the definition(s) of output parameters to save.
                simulation/  # The simulation classes for running different coral configurations.
            vegetation/ # Module containing the different vegetation model definitions.
                bioprocess/ # Contains all the different bio processes required by the vegetation model.
                model/  # Defines the vegetation model and related (such as constants).
                output/ # Contains the definition(s) of output parameters to save.
                simulation/  # The simulation classes for running different vegetation configurations.
        core/           # Module containing all the classes and methods of NBSDynamics.
            biota/      # The base class biota from which sub models can inherit.
            hydrodynamics/ # The different hydrodynamic models ('Transect', 'Delft3D', 'Reef0D', 'Reef1D')
            output/     # The base output models.
            simulation/ # The base simulation classes.        
        tools/          # Module containing tools used by the coral_model module.
    test/               # Module based on pytest to mirror and test all the classes from src/

## Version control
Currently versioning is done with the help of [`commitizen`](https://commitizen-tools.github.io/commitizen/) using a tag system of v.Major.Minor.Patch . A the [`changelog.md`](changelog.md) is generated automatically.

## Project architecture
At the moment of this edition (version v.0.5.1), the project has undergone several refactorings and is divided in different modules and components as explained in the 'Project layout' section.

A summary of the architecture can be seen in the following 'reduced' class diagram:
![`Class diagram`](./diagrams/general_class_diagram-GeneralClassDiagram.png)

### SimulationProtocol
![`SimulationProtocol`](./diagrams/general_class_diagram-SimulationProtocol.png)

### HydrodynamicProtocol
![`HydrodynamicProtocol`](./diagrams/general_class_diagram-HydrodynamicProtocol.png)

### CoralProtocol
![`CoralProtocol`](./diagrams/general_class_diagram-CoralProtocol.png)

### OutputProtocol
![`OutputProtocol`](./diagrams/general_class_diagram-OutputProtocol.png)
