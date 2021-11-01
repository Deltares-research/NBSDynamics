import pytest
from pathlib import Path
from src.coral_model.core import Coral
from src.coral_model.loop import Simulation

from test.utils import TestUtils


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

    @pytest.mark.skip(reason="Not yet supported.")
    def test_given_interface_transect_case_runs(self):
        model_dir = TestUtils.get_local_test_data_dir("Mariya_model")
        # define the basic Simulation object, indicating already here the type of hydrodynamics
        run_trans = Simulation(mode="Transect")
        # set the working directory and its subdirectories (input, output, figures)
        run_trans.set_directories(model_dir / "Run_Transect")
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
        coral = Coral(run_trans.constants, 0.1, 0.1, 0.05, 0.05, 0.2, 1.0)
        coral = run_trans.initiate(coral)
        # simulation
        run_trans.exec(coral)
        # finalizing
        run_trans.finalise()
        # done

    def test_given_interface_transect_new_case_runs(self):
        """
        coral_model - interface

        @author: Gijs G. Hendrickx

        """
        test_dir = TestUtils.get_local_test_data_dir("transect_case")
        assert test_dir.is_dir()

        # define the basic Simulation object, indicating already here the type of hydrodynamics
        run_trans = Simulation(mode="Transect")
        # set the working directory and its subdirectories (input, output, figures)
        run_trans.set_directories(test_dir / "Run_26_10_massive")
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
        coral = Coral(run_trans.constants, 0.125, 0.125, 0.1, 0.1, 0.2, 0.6)

        coral = run_trans.initiate(coral)
        # simulation
        run_trans.exec(coral)
        # finalizing
        run_trans.finalise()
        # done

        assert (run_trans.output_dir / "CoralModel_his.nc").is_file()
        assert (run_trans.output_dir / "CoralModel_map.nc").is_file()
