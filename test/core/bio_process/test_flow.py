import pytest

from src.core.bio_process.flow import Flow


class Testflow:
    @pytest.mark.parametrize("wac_type", [pytest.param(None), pytest.param("unknown")])
    def test_wave_attenuation_unknown_type_raises_valueerror(self, wac_type: str):
        input_dict = dict(
            constants=None,
            diameter=None,
            height=None,
            distance=None,
            velocity=None,
            period=None,
            depth=None,
            wac_type=wac_type,
        )
        with pytest.raises(ValueError) as e_info:
            Flow.wave_attenuation(**input_dict)
        assert (
            str(e_info.value)
            == f"WAC-type ({wac_type}) not in dict_keys(['current', 'wave'])."
        )

    @pytest.mark.parametrize(
        "depth, height",
        [
            pytest.param(1, 1, id="Equal values"),
            pytest.param(1, 2, id="Height greater"),
        ],
    )
    @pytest.mark.parametrize(
        "wac_type", [pytest.param("current"), pytest.param("wave")]
    )
    def test_depth_less_or_equal_height_returns_wac(
        self, depth: float, height: float, wac_type: str
    ):
        class test_constants:
            Cs = 0.42

        input_dict = dict(
            constants=test_constants(),
            diameter=4.2,
            height=height,
            distance=2.4,
            velocity=None,
            period=None,
            depth=depth,
            wac_type=wac_type,
        )

        assert Flow.wave_attenuation(**input_dict) == 1.0

    def test_wave_attenuation_current_returns_wac(self):
        class test_constants:
            Cs = 2.4
            maxiter_k = 1
            maxiter_aw = 1
            Cm = 4.2
            psi = 0.24
            nu = 1
            err = 0.01
            numericTheta = 0.24

        input_dict = dict(
            constants=test_constants(),
            diameter=0.42,
            height=2,
            distance=24,
            velocity=0.2,
            period=0.4,
            depth=4,
            wac_type="current",
        )

        assert Flow.wave_attenuation(**input_dict) == pytest.approx(0.9888712497166652)
