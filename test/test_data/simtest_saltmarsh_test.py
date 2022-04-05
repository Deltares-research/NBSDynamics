from typing import Tuple

import numpy as np
import pytest

from pathlib import Path
from src.core.simulation.veg_base_simulation import BaseSimulation
from test.utils import TestUtils  # check why I need to go to test folder to work
from src.core.simulation.veg_delft3d_simulation import VegFlowFmSimulation
from src.core.common.constants_veg import Constants
from src.core.vegetation.veg_model import Vegetation

# test_dir = TestUtils.get_local_test_data_dir("delft3d_case")
test_dir = Path(
    r"C:\Users\dzimball\PycharmProjects\NBSDynamics\test\test_data\saltmarsh_test_Pdrive\standardmodel_stijn")
# dll_repo = TestUtils.get_external_repo("DimrDllDependencies")
dll_repo = Path(r"C:\Users\dzimball\dimrdlldependencies")
# dll_repo= Path (r"c:\Program Files (x86)\Deltares\Delft3D Flexible Mesh Suite HMWQ (2021.03)\plugins\DeltaShell.Dimr")
assert test_dir.is_dir()
kernels_dir = dll_repo / "kernels" / "x64"
# test_case = dll_repo / "test_cases" / "c01_test1_smalltidalbasin_vegblock"

input_dir = test_dir  # / "input"

sim_run = VegFlowFmSimulation(
    working_dir=test_dir,
    constants=Constants(species="Spartina anglica"),
    # constants=input_dir/ "MinFiles" / "fm" / "veg.ext",

    hydrodynamics=dict(
        working_dir=test_dir / "d3d_work",
        d3d_home=kernels_dir,
        dll_path=kernels_dir / "dflowfm_with_shared" / "bin" / "dflowfm.dll",
        # definition_file=test_case / "fm" / "shallow_wave.mdu",
        definition_file=input_dir / "FlowFM.mdu",

    ),
    output=dict(
        output_dir=test_dir / "output",
        map_output=dict(output_params=dict(fme=False)),
        his_output=dict(
            # xy_stations=np.array([[0, 0], [1, 1]]),
            output_params=dict(fme=False),
        ),

    ),
    veg=Vegetation(species="Spartina anglica")
)

# Run simulation.
sim_run.initiate()
sim_run.run(5)
sim_run.finalise()
