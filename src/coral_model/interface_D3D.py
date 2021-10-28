"""
coral_model - interface

@author: Gijs G. Hendrickx
@contributor: Peter M.J. Herman
"""
import os
from time import sleep

from coral_model.core import Coral
from coral_model.loop import Simulation

base_dir = os.path.join("C:\\", "Users", "herman", "CoralModel")
# define the basic Simulation object, indicating already here the type of hydrodynamics
run001 = Simulation(mode="Delft3D")
# set the working directory and its subdirectories (input, output, figures)
run001.set_directories(os.path.join(base_dir, "work_dir_example"))
# read the input file with parameters (processes, parameters,constants, now all in "constants")
run001.read_parameters(file="coral_input.txt", folder=run001.input_dir)
# environment definition
run001.environment.from_file("light", "TS_PAR.txt", folder=run001.input_dir)
run001.environment.from_file("temperature", "TS_SST.txt", folder=run001.input_dir)

# hydrodynamic model
run001.hydrodynamics.working_dir = os.path.join(run001.working_dir, "d3d_work")
run001.hydrodynamics.d3d_home = os.path.join(base_dir, "d3d_suite")
run001.hydrodynamics.mdu = os.path.join("fm", "FlowFM.mdu")
run001.hydrodynamics.config = "dimr_config.xml"
run001.hydrodynamics.set_update_intervals(300, 300)
# sleep(2)
run001.hydrodynamics.initiate()

# check
print(run001.hydrodynamics.settings)
# define output
run001.define_output("map", fme=False)
run001.define_output("his", fme=False)
run001.output.xy_stations = (0, 0)
# initiate coral
coral = Coral(run001.constants, 0.1, 0.1, 0.05, 0.05, 0.2)
print("coral defined")
coral = run001.initiate(coral)


print("coral initiated")

# simulation
run001.exec(coral)

# finalizing
run001.finalise()

# done
