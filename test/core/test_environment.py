from datetime import datetime
from pathlib import Path
from test.utils import TestUtils
from typing import Any, Iterable, List, Union

import numpy as np
import pandas as pd
import pytest

from src.core.environment import Environment

daily_params: List[str] = [
    ("light"),
    ("light_attenuation"),
    ("temperature"),
    ("aragonite"),
]

unsupported_params = [(None), (42.24), (42), ("not/a/Path"), datetime(2021, 12, 20)]


def env_params_cases() -> List[pytest.param]:
    input_dir = TestUtils.get_local_test_file("transect_case") / "input"
    dp_file = input_dir / "TS_PAR.txt"
    assert dp_file.is_file()
    env_params = [pytest.param(d_p, dp_file, id=d_p) for d_p in daily_params]
    sc_file = input_dir / "TS_stormcat.txt"
    sc_param = pytest.param("storm_category", sc_file, id="storm_category")
    assert sc_file.is_file()
    env_params.append(sc_param)
    return env_params


class TestEnvironment:
    @staticmethod
    def get_test_input_data() -> Path:
        return TestUtils.get_local_test_file("transect_case") / "input"

    def test_init_environment(self):
        test_env = Environment()
        assert test_env.dates is not None
        assert test_env.light is None
        assert test_env.light_attenuation is None
        assert test_env.temperature is None
        assert test_env.aragonite is None
        assert test_env.storm_category is None

    @pytest.mark.parametrize("input_key, input_file", env_params_cases())
    def test_init_environment_with_key_and_file(self, input_key: str, input_file: Path):
        input_dict = {input_key: input_file}
        test_env = Environment(**input_dict)
        assert isinstance(test_env.dict()[input_key], pd.DataFrame)

    @pytest.mark.parametrize("input_key, _", env_params_cases())
    def test_init_environment_with_key_and_dataframe(self, input_key: str, _: Path):
        # We are not going to use input_file, but the test case applies, so we reuse it.
        test_value = pd.DataFrame(np.array([[42, 24], [24, 42]]), columns=["a", "b"])
        input_dict = {input_key: test_value}
        test_env = Environment(**input_dict)
        assert all(test_env.dict()[input_key] == test_value)

    def test_validate_dataframe_or_path_from_file(self):
        test_file = self.get_test_input_data() / "TS_PAR.txt"
        assert test_file.is_file()
        return_value = Environment.validate_dataframe_or_path(test_file)
        assert isinstance(return_value, pd.DataFrame)

    def test_validate_dataframe_or_path_from_dataframe(self):
        test_value = pd.DataFrame(np.array([[42, 24], [24, 42]]), columns=["a", "b"])
        test_env = Environment.validate_storm_category(test_value)
        assert all(test_env == test_value)

    @pytest.mark.parametrize("unsupported", unsupported_params)
    def test_validate_dataframe_or_path_from_unsupported_raises(self, unsupported: Any):
        with pytest.raises(NotImplementedError) as e_err:
            Environment.validate_dataframe_or_path(unsupported)
        assert (
            str(e_err.value) == f"Validator not available for type {type(unsupported)}"
        )

    @pytest.mark.parametrize("unsupported", unsupported_params)
    def test_validate_storm_category_from_unsupported_raises(self, unsupported: Any):
        with pytest.raises(NotImplementedError) as e_err:
            Environment.validate_storm_category(unsupported)
        assert (
            str(e_err.value) == f"Validator not available for type {type(unsupported)}"
        )

    @pytest.mark.parametrize("start_date", [("2001, 10, 02"), (datetime(2001, 10, 2))])
    @pytest.mark.parametrize("end_date", [("2021, 12, 21"), (datetime(2021, 12, 21))])
    def test_set_dates_update_dataframe(
        self, start_date: Union[str, datetime], end_date: Union[str, datetime]
    ):
        test_env = Environment()
        initial_dates = test_env.get_dates()

        test_env.set_dates(start_date, end_date)
        new_dates = test_env.get_dates()
        assert isinstance(test_env.dates, pd.DataFrame)
        assert initial_dates.array[0] != new_dates.array[0]
        assert initial_dates.array[-1] != new_dates.array[-1]

    @pytest.mark.parametrize("start_date", [("2001, 10, 02"), (datetime(2001, 10, 2))])
    @pytest.mark.parametrize("end_date", [("2021, 12, 21"), (datetime(2021, 12, 21))])
    def test_get_dates_dataframe(
        self, start_date: Union[str, datetime], end_date: Union[str, datetime]
    ):
        test_value = Environment.get_dates_dataframe(start_date, end_date)
        assert isinstance(test_value, pd.DataFrame)

    def test_validate_dates_as_dataframe(self):
        test_value = pd.date_range("2021, 01, 01", "2021, 12, 21")
        test_value = pd.DataFrame({"date": test_value})
        validated_dates = Environment.prevalidate_dates(test_value)
        assert all(validated_dates == test_value)

    @pytest.mark.parametrize(
        "it_value",
        [
            pytest.param(("2021, 01, 01", "2021, 12, 21"), id="As tuple"),
            pytest.param(["2021, 01, 01", "2021, 12, 21"], id="As list"),
        ],
    )
    def test_validate_dates_as_iterable(self, it_value: Iterable):
        return_value = Environment.prevalidate_dates(it_value)
        assert isinstance(return_value, pd.DataFrame)

    @pytest.mark.parametrize("unsupported", unsupported_params)
    def test_validate_dates_from_unsupported_raises(self, unsupported: Any):
        with pytest.raises(NotImplementedError) as e_err:
            Environment.validate_storm_category(unsupported)
        assert (
            str(e_err.value) == f"Validator not available for type {type(unsupported)}"
        )
