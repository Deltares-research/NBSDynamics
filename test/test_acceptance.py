from pathlib import Path

import netCDF4
from test.utils import TestUtils

import pytest

from src.coral_model.core import Coral
from src.coral_model.loop import Simulation
from netCDF4 import Dataset


class TestAcceptance:
    @pytest.mark.skip(reason="Not yet supported.")
    def test_given_interface_d3d_case_runs(self):
        # Test based on interface_D3D.py
        model_dir = TestUtils.get_local_test_data_dir("CoralModel")
        assert model_dir.is_dir()

        fm_dir = TestUtils.get_external_test_data_dir("fm")
        assert fm_dir.is_dir()

        sim_run = Simulation(mode="Delft3D")

        # set the working directory and its subdirectories (input, output, figures)
        sim_run.set_directories(model_dir / "work_dir_example")
        # read the input file with parameters (processes, parameters,constants, now all in "constants")
        sim_run.read_parameters(file="coral_input.txt", folder=sim_run.input_dir)
        # environment definition
        sim_run.environment.from_file("light", "TS_PAR.txt", folder=sim_run.input_dir)
        sim_run.environment.from_file(
            "temperature", "TS_SST.txt", folder=sim_run.input_dir
        )

        # hydrodynamic model
        sim_run.hydrodynamics.working_dir = sim_run.working_dir / "d3d_work"
        sim_run.hydrodynamics.d3d_home = model_dir / "d3d_suite"
        sim_run.hydrodynamics.mdu = fm_dir / "FlowFM.mdu"
        sim_run.hydrodynamics.config = "dimr_config.xml"
        sim_run.hydrodynamics.set_update_intervals(300, 300)
        # sleep(2)
        sim_run.hydrodynamics.initiate()

        # check
        print(sim_run.hydrodynamics.settings)
        # define output
        sim_run.define_output("map", fme=False)
        sim_run.define_output("his", fme=False)
        sim_run.output.xy_stations = (0, 0)
        # initiate coral
        coral = Coral(sim_run.constants, 0.1, 0.1, 0.05, 0.05, 0.2)
        print("coral defined")
        coral = sim_run.initiate(coral)

        print("coral initiated")
        # simulation
        sim_run.exec(coral)

        # finalizing
        sim_run.finalise()

    @pytest.mark.parametrize(
        "coral_values",
        [
            pytest.param(
                dict(dc=0.1, hc=0.1, bc=0.05, tc=0.05, ac=0.2, species_constant=1.0),
                id="Species constant 1.0",
            ),
            pytest.param(
                dict(dc=0.125, hc=0.125, bc=0.1, tc=0.1, ac=0.2, species_constant=0.6),
                id="Species constant 0.6",
            ),
        ],
    )
    def test_given_interface_transect_runs(self, coral_values: dict):
        # 1. Define test data.
        test_dir = TestUtils.get_local_test_data_dir("transect_case")
        assert test_dir.is_dir()
        working_dir = test_dir / "Run_26_10_massive"

        # 1.b. Remove all output data in case it exists from previous runs.
        output_dir = working_dir / "output"
        his_output_file = output_dir / "CoralModel_his.nc"
        map_output_file = output_dir / "CoralModel_map.nc"
        if his_output_file.exists():
            his_output_file.unlink()
        if map_output_file.exists():
            map_output_file.unlink()

        # 2. Prepare model.
        # Define the basic Simulation object, indicating already here the type of hydrodynamics
        run_trans = Simulation(mode="Transect")
        run_trans.set_directories(working_dir)
        # read the input file with parameters (processes, parameters,constants, now all in "constants")
        run_trans.read_parameters(file="coral_input.txt", folder=run_trans.input_dir)
        # environment definition
        run_trans.environment.from_file(
            "light", "TS_PAR.txt", folder=run_trans.input_dir
        )
        run_trans.environment.from_file(
            "temperature", "TS_SST.txt", folder=run_trans.input_dir
        )
        run_trans.environment.from_file(
            "storm", "TS_stormcat2.txt", folder=run_trans.input_dir
        )

        # time definition
        run_trans.environment.set_dates(start_date="2000-01-01", end_date="2100-01-01")

        # hydrodynamic model
        # settings for a 1D idealized transect using fixed currents and Soulsby
        # orbital velocities depending on stormcat and depth
        run_trans.hydrodynamics.working_dir = run_trans.working_dir
        run_trans.hydrodynamics.mdu = Path("input") / "TS_waves.txt"
        run_trans.hydrodynamics.config = Path("input") / "config.csv"
        run_trans.hydrodynamics.initiate()
        # check
        print(run_trans.hydrodynamics.settings)
        # define output
        run_trans.define_output("map", fme=False)
        run_trans.define_output("his", fme=False)
        # initiate coral
        coral_dict = {**dict(constants=run_trans.constants), **coral_values}
        coral = Coral(**coral_dict)
        coral = run_trans.initiate(coral)

        # 3. Run simulation
        run_trans.exec(coral)
        run_trans.finalise()

        # 4. Verify expectations.
        output_his_file = run_trans.output_dir / "CoralModel_his.nc"
        assert output_his_file.exists()
        output_map_file = run_trans.output_dir / "CoralModel_map.nc"
        assert output_map_file.exists()

    @pytest.mark.skip(reason="Only to run locally.")
    def test_compare_netcdf_manually(self):
        output_dir = (
            TestUtils.get_local_test_data_dir("transect_case")
            / "Run_26_10_massive"
            / "output"
        )
        expected_dir = (
            TestUtils.get_local_test_data_dir("transect_case") / "expected" / "output"
        )

        def get_output_netcdf(output_dir: Path, filename: str) -> Dataset:
            nc_filepath = output_dir / filename
            assert nc_filepath.is_file()
            return Dataset(nc_filepath, "r", format="NETCDF4")

        def compare_map_netcdf(map_file: str):
            expected_map = get_output_netcdf(expected_dir, map_file)
            output_map = get_output_netcdf(output_dir, map_file)
            assert expected_map == output_map
            expected_map.close()
            output_map.close()

        def compare_his_netcdf(his_file: str):
            expected_his = get_output_netcdf(expected_dir, his_file)
            output_his = get_output_netcdf(output_dir, his_file)
            assert expected_his == output_his
            expected_his.close()
            output_his.close()

        compare_map_netcdf("CoralModel_map.nc")
        compare_his_netcdf("CoralModel_his.nc")
