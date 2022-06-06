from test.utils import TestUtils
from src.biota_models.vegetation.model.veg_constants import VegetationConstants
from src.biota_models.vegetation.model.veg_model import Vegetation
from src.biota_models.vegetation.simulation.veg_delft3d_simulation import (
    VegFlowFmSimulation,
)


# 1. Define attributes
test_dir = TestUtils.get_local_test_data_dir("FlowFM")
# We need to specify where the DIMR directory (WITH ALL THE SHARED BINARIES) is located.
dll_repo = TestUtils.get_external_repo("code_23062020")
kernels_dir = dll_repo / "x64"
output_dir = test_dir / "output2y"
test_case = test_dir / "input"
species = "Puccinellia"

# 2. Prepare model.
# Create vegetation constants based on the above species.
veg_constants = VegetationConstants(species=species)
sim_run = VegFlowFmSimulation(
    working_dir=test_dir,
    constants=veg_constants,
    hydrodynamics=dict(
        working_dir=test_dir / "d3d_work",
        d3d_home=kernels_dir,
        dll_path=kernels_dir / "dflowfm_with_shared" / "bin" / "dflowfm.dll",
        definition_file=test_case / "fm" / "FlowFM.mdu",
    ),
    output=dict(
        output_dir=output_dir,
        map_output=dict(output_params=dict()),
        his_output=dict(
            output_params=dict(),
        ),
    ),
    biota=Vegetation(species=species, constants=veg_constants),
)

# 3. Run simulation.
sim_run.initiate()
sim_run.run(2)
sim_run.finalise()