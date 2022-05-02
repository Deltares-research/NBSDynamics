from src.core.hydrodynamics.delft3d import Delft3D
from src.core.hydrodynamics.delft3d import FlowFmModel
from pathlib import Path
from test.utils import TestUtils
import numpy as np

test_dir = TestUtils.get_local_test_data_dir("sm_testcase6")
dll_repo = TestUtils.get_external_repo("DimrDllDependencies")
kernels_dir = dll_repo / "kernels" / "x64"

test_case = test_dir / "input" / "MinFiles"

hydro = FlowFmModel(
                working_dir=test_dir / "d3d_work",
                d3d_home=kernels_dir,
                dll_path=kernels_dir / "dflowfm_with_shared" / "bin" / "dflowfm",
                definition_file=test_case / "fm" / "test_case6.mdu")


hydro.initiate()

veg_den = np.zeros((hydro.space,1))
veg_height = np.zeros((hydro.space,1))
veg_stemdia = np.zeros((hydro.space,1))

fill = [69,70,71,89,90,91,109,110,111,129,130,131]

for i in fill:
    veg_den[i]= 420
    veg_height[i]= 1.3
    veg_stemdia[i]= 0.005


hydro.set_variable(
    "rnveg", veg_den
)  # [1/m2] 3D plant density , 2D part is basis input (1/m2)
hydro.set_variable(
    "diaveg", veg_stemdia
)  # [m] 3D plant diameter, 2D part is basis input (m)
hydro.set_variable(
    "stemheight", veg_height
)  # [m] 2D plant heights (m)

hydro.model_wrapper.update(dt=32000000)
