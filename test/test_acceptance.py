import platform
from pathlib import Path
from test.utils import TestUtils
from typing import List

import numpy
import pytest
from netCDF4 import Dataset
from numpy import loadtxt, savetxt
from numpy.ma import getdata
from numpy.ma.core import var

from src.core.coral_model import Coral
from src.core.loop import Simulation
from src.tools.plot_output import plot_his, plot_map


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

    def test_given_interface_transect_runs(self):
        # 1. Define test data.
        test_dir = TestUtils.get_local_test_data_dir("transect_case")
        assert test_dir.is_dir()

        # 1.b. Remove all output data in case it exists from previous runs.
        his_filename = "CoralModel_his.nc"
        map_filename = "CoralModel_map.nc"
        output_dir = test_dir / "output"

        his_output_file = output_dir / his_filename
        his_output_file.unlink(missing_ok=True)

        map_output_file = output_dir / map_filename
        map_output_file.unlink(missing_ok=True)

        # 2. Prepare model.
        # Define the basic Simulation object, indicating already here the type of hydrodynamics
        run_trans = Simulation(mode="Transect")
        run_trans.set_directories(test_dir)
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
        coral_dict = {
            **dict(constants=run_trans.constants),
            **dict(dc=0.125, hc=0.125, bc=0.1, tc=0.1, ac=0.2, species_constant=0.6),
        }
        coral = Coral(**coral_dict)
        coral = run_trans.initiate(coral)

        # 3. Run simulation
        run_trans.exec(coral)
        run_trans.finalise()

        # 4. Verify expectations.
        expected_dir = (
            TestUtils.get_local_test_data_dir("transect_case") / "expected_output"
        )

        def compare_files(created_file: Path):
            def normalize_name_with_capitals(name: str) -> str:
                if any(x.isupper() for x in name):
                    return name + name
                return name

            ref_dir = expected_dir / created_file.stem.split("_")[-1]
            with Dataset(created_file, "r", format="NETCDF4") as out_netcdf:
                for variable in out_netcdf.variables:
                    variable_filename = normalize_name_with_capitals(variable)
                    expected_file = (
                        ref_dir / f"ref_{variable_filename}_{created_file.stem}.txt"
                    )
                    assert (
                        expected_file.is_file()
                    ), f"Expected file for variable {variable} not found at {expected_file}"
                    ref_variable = loadtxt(expected_file)
                    assert numpy.allclose(
                        ref_variable, out_netcdf[variable]
                    ), f"{variable} not close to reference data."

        compare_files(run_trans.output_dir / his_filename)
        compare_files(run_trans.output_dir / map_filename)

        # 5. Verify plotting can be done.
        if "win" not in platform.system().lower():
            return
        plot_his(run_trans.output_dir / his_filename)
        plot_map(run_trans.output_dir / map_filename)

    @pytest.mark.skip(reason="Only to be run locally.")
    @pytest.mark.parametrize(
        "nc_filename",
        [
            pytest.param("CoralModel_his.nc", id="His variables"),
            pytest.param("CoralModel_map.nc", id="Map variables"),
        ],
    )
    def test_util_output_variables_netcdf(self, nc_filename: str):
        """
        This test is only meant to be run locally, it helps generating the expected data as .txt files.
        """
        expected_dir = (
            TestUtils.get_local_test_data_dir("transect_case") / "expected_output"
        )

        def normalize_name_with_capitals(name: str) -> str:
            if any(x.isupper() for x in name):
                return name + name
            return name

        def output_file(netcdf_file: Path):
            out_dir = netcdf_file.parent / netcdf_file.stem.split("_")[-1]
            out_dir.mkdir(parents=True, exist_ok=True)
            with Dataset(netcdf_file, "r", format="NETCDF4") as ref_netcdf:
                for variable in ref_netcdf.variables:
                    variable_name = normalize_name_with_capitals(variable)
                    ref_file = out_dir / f"ref_{variable_name}_{netcdf_file.stem}.txt"
                    savetxt(ref_file, ref_netcdf[variable])

        output_file(expected_dir / nc_filename)

    @pytest.mark.skip(reason="Only to run locally.")
    def test_plot_map_output(self):
        expected_file = (
            TestUtils.get_local_test_data_dir("transect_case")
            / "output"
            / "CoralModel_map.nc"
        )

        plot_map(expected_file)

    @pytest.mark.skip(reason="Only to run locally.")
    def test_plot_his_output(self):
        expected_file = (
            TestUtils.get_local_test_data_dir("transect_case")
            / "output"
            / "CoralModel_his.nc"
        )

        plot_his(expected_file)
