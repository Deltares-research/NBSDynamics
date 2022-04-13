from test.coral.bio_process.bio_utils import matrix_1x1, matrix_2x2

import pytest

from src.coral.model.coral_constants import CoralConstants
from src.coral.bio_process.flow import Flow
from src.core.common.space_time import DataReshape


class TestFlow:
    def test_init_flow(self, matrix_1x1):
        flow = Flow(0.1, 0.1, 5, 4)
        assert flow.uc[0] == 0.1
        assert flow.uw[0] == 0.1
        assert flow.h == 5
        assert flow.Tp[0] == 4

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

    def test_wave_attenuation(self):
        # input array
        diameter = [0.1, 0.2, 0.4]  # [m]
        height = 0.3  # [m]
        distance = [0.3, 0.4, 0.6]  # [m]
        velocity = 0.05  # [m s-1]
        period = 4  # [s]
        depth = 0.75  # [m]

        # answers
        answer = [
            0.73539733818684030,
            0.47628599416211803,
            0.20277038395777466,
        ]

        for i in range(3):
            wac = Flow.wave_attenuation(
                constants=CoralConstants(),
                diameter=diameter[i],
                height=height,
                distance=distance[i],
                velocity=velocity,
                period=period,
                depth=depth,
                wac_type="wave",
            )
            assert float(wac), pytest.approx(answer[i], 0.1)


class TestFlox2x2:
    def test_initiation(self, matrix_2x2: DataReshape):
        flow = Flow([0.1, 0.1], [0.1, 0.1], [5, 5], [4, 4])
        for i in range(matrix_2x2.space):
            assert float(flow.uc[i]) == 0.1
            assert float(flow.uw[i]) == 0.1
            for j in range(matrix_2x2.time):
                assert float(flow.h[i, j]) == 5
            assert float(flow.Tp[i]) == 4
