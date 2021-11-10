import numpy as np
from test.utils import TestUtils
import pytest
from pathlib import Path
from src.core.environment import Environment
import pandas as pd
from typing import Any, List

daily_params: List[str] = [
    ("light"),
    ("light_attenuation"),
    ("temperature"),
    ("aragonite"),
]


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

    @pytest.mark.parametrize("unsupported", [(None), (float), (int), ("not/a/Path")])
    def test_validate_dataframe_or_path_from_unsupported_raises(self, unsupported: Any):
        with pytest.raises(NotImplementedError) as e_err:
            Environment.validate_dataframe_or_path(unsupported)
        assert (
            str(e_err.value) == f"Validator not available for type {type(unsupported)}"
        )

    @pytest.mark.parametrize("unsupported", [(None), (float), (int), ("not/a/Path")])
    def test_validate_storm_category_from_unsupported_raises(self, unsupported: Any):
        with pytest.raises(NotImplementedError) as e_err:
            Environment.validate_storm_category(unsupported)
        assert (
            str(e_err.value) == f"Validator not available for type {type(unsupported)}"
        )
