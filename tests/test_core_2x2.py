import unittest

from CoralModel_v3 import core
from CoralModel_v3.core import Coral, Light, Flow, Temperature, PROCESSES, Photosynthesis, \
    Calcification, Morphology, Dislodgement, Recruitment
from CoralModel_v3.utils import DataReshape


core.RESHAPE = DataReshape((2, 2))


class TestCoral(unittest.TestCase):

    # TODO: Set all tested variables as floats; hard-coded.

    def test_default(self):
        coral = Coral(.2, .3, .1, .15, .3)
        self.assertEqual(coral.Csp, 1)

    def test_input_single(self):
        coral = Coral(.2, .3, .1, .15, .3)
        self.assertEqual(coral.dc, .2)
        self.assertEqual(coral.hc, .3)
        self.assertEqual(coral.bc, .1)
        self.assertEqual(coral.tc, .15)
        self.assertEqual(coral.ac, .3)

    def test_input_multiple(self):
        coral = Coral([.2, .2], [.3, .3], [.1, .1], [.15, .15], [.3, .3])
        for i in range(core.RESHAPE.space):
            self.assertEqual(coral.dc[i], .2)
            self.assertEqual(coral.hc[i], .3)
            self.assertEqual(coral.bc[i], .1)
            self.assertEqual(coral.tc[i], .15)
            self.assertEqual(coral.ac[i], .3)

    def test_auto_initiate1(self):
        coral = Coral(.2, .3, .1, .15, .3)
        coral.initiate_spatial_morphology()
        for i in range(core.RESHAPE.space):
            self.assertEqual(coral.dc[i], .2)
            self.assertEqual(coral.hc[i], .3)
            self.assertEqual(coral.bc[i], .1)
            self.assertEqual(coral.tc[i], .15)
            self.assertEqual(coral.ac[i], .3)

    def test_auto_initiate2(self):
        coral = Coral(.2, .3, .1, .15, .3)
        coral.initiate_spatial_morphology(cover=[1, 1])
        for i in range(core.RESHAPE.space):
            self.assertEqual(coral.dc[i], .2)
            self.assertEqual(coral.hc[i], .3)
            self.assertEqual(coral.bc[i], .1)
            self.assertEqual(coral.tc[i], .15)
            self.assertEqual(coral.ac[i], .3)

    def test_auto_initiate3(self):
        coral = Coral(.2, .3, .1, .15, .3)
        coral.initiate_spatial_morphology(cover=[1, 0])
        self.assertEqual(coral.dc[0], .2)
        self.assertEqual(coral.hc[0], .3)
        self.assertEqual(coral.bc[0], .1)
        self.assertEqual(coral.tc[0], .15)
        self.assertEqual(coral.ac[0], .3)
        self.assertEqual(coral.dc[1], 0)
        self.assertEqual(coral.hc[1], 0)
        self.assertEqual(coral.bc[1], 0)
        self.assertEqual(coral.tc[1], 0)
        self.assertEqual(coral.ac[1], 0)

    def test_representative_diameter(self):
        coral = Coral(.2, .3, .1, .15, .3)
        coral.initiate_spatial_morphology()
        for i in range(core.RESHAPE.space):
            self.assertAlmostEqual(float(coral.dc_rep[i]), .15)

    def test_morphological_ratios(self):
        coral = Coral(.2, .3, .1, .15, .3)
        coral.initiate_spatial_morphology()
        for i in range(core.RESHAPE.space):
            self.assertAlmostEqual(float(coral.rf[i]), 1.5)
            self.assertAlmostEqual(float(coral.rp[i]), .5)
            self.assertAlmostEqual(float(coral.rs[i]), 2/3)

    def test_coral_volume(self):
        coral = Coral(.2, .3, .1, .15, .3)
        coral.initiate_spatial_morphology()
        for i in range(core.RESHAPE.space):
            self.assertAlmostEqual(float(coral.volume[i]), .0058904862254808635)

    def test_vegetation_density(self):
        coral = Coral(.2, .3, .1, .15, .3)
        coral.initiate_spatial_morphology()
        for i in range(core.RESHAPE.space):
            self.assertAlmostEqual(coral.as_vegetation_density[i], 3.3333333333333335)

    def test_coral_cover(self):
        coral = Coral(.2, .3, .1, .15, .3)
        coral.initiate_spatial_morphology()
        for i in range(core.RESHAPE.space):
            self.assertEqual(coral.cover[i], 1)


class TestLight(unittest.TestCase):

    def test_initiation(self):
        light = Light([600, 600], [.1, .1], [5, 5])
        for i in range(core.RESHAPE.space):
            for j in range(core.RESHAPE.time):
                self.assertEqual(float(light.I0[i, j]), 600)
                self.assertEqual(float(light.Kd[i, j]), .1)
                self.assertEqual(float(light.h[i, j]), 5)

    def test_representative_light(self):
        light = Light([600, 600], [.1, .1], [5, 5])
        # base light
        coral = Coral(.2, .3, .1, .15, .3)
        coral.initiate_spatial_morphology()
        light.rep_light(coral)
        answer = 217.0490558
        for i in range(core.RESHAPE.space):
            for j in range(core.RESHAPE.time):
                self.assertAlmostEqual(float(coral.light[i, j]), answer)
        # no base light
        coral = Coral(.4, .3, .2, .15, .3)
        coral.initiate_spatial_morphology()
        light.rep_light(coral)
        answer = 253.8318634
        for i in range(core.RESHAPE.space):
            for j in range(core.RESHAPE.time):
                self.assertAlmostEqual(float(coral.light[i, j]), answer)

    def test_coral_biomass(self):
        light = Light([600, 600], [.1, .1], [5, 5])
        # base light
        coral = Coral(.2, .3, .1, .15, .3)
        coral.initiate_spatial_morphology()
        light.biomass(coral)
        answer = .14287642
        for i in range(core.RESHAPE.space):
            for j in range(core.RESHAPE.time):
                self.assertAlmostEqual(float(coral.Bc[i, j]), answer)
        # no base light
        coral = Coral(.4, .3, .2, .15, .3)
        coral.initiate_spatial_morphology()
        light.biomass(coral)
        answer = .314159265
        for i in range(core.RESHAPE.space):
            for j in range(core.RESHAPE.time):
                self.assertAlmostEqual(float(coral.Bc[i, j]), answer)

    def test_base_light(self):
        light = Light([600, 600], [.1, .1], [5, 5])
        # base light
        coral = Coral(.2, .3, .1, .15, .3)
        coral.initiate_spatial_morphology()
        result = light.base_light(coral)
        answer = .05478977
        for i in range(core.RESHAPE.space):
            for j in range(core.RESHAPE.time):
                self.assertAlmostEqual(float(result[i, j]), answer)
        # no base light
        coral = Coral(.4, .3, .2, .15, .3)
        coral.initiate_spatial_morphology()
        result = light.base_light(coral)
        answer = 0
        for i in range(core.RESHAPE.space):
            for j in range(core.RESHAPE.time):
                self.assertAlmostEqual(float(result[i, j]), answer)


class TestFlow(unittest.TestCase):

    def test_initiation(self):
        flow = Flow([.1, .1], [.1, .1], [5, 5], [4, 4])
        for i in range(core.RESHAPE.space):
            self.assertEqual(float(flow.uc[i]), .1)
            self.assertEqual(float(flow.uw[i]), .1)
            for j in range(core.RESHAPE.time):
                self.assertEqual(float(flow.h[i, j]), 5)
            self.assertEqual(float(flow.Tp[i]), 4)


class TestTemperature(unittest.TestCase):

    def test_initiation(self):
        temperature = Temperature([300, 300])
        for i in range(core.RESHAPE.space):
            for j in range(core.RESHAPE.time):
                self.assertEqual(temperature.T[i, j], 300)

    def test_coral_temperature(self):
        temperature = Temperature([300, 300])
        coral = Coral(.2, .3, .1, .15, .3)
        coral.initiate_spatial_morphology()
        coral.delta_t = [.001, .001]
        coral.light = [600, 600]
        temperature.coral_temperature(coral)
        for i in range(core.RESHAPE.space):
            for j in range(core.RESHAPE.time):
                self.assertAlmostEqual(float(coral.temp[i, j]), 300.00492692)

    def test_no_tme(self):
        PROCESSES.tme = False
        temperature = Temperature([300, 300])
        coral = Coral(.2, .3, .1, .15, .3)
        coral.initiate_spatial_morphology()
        coral.delta_t = [.001, .001]
        coral.light = [600, 600]
        temperature.coral_temperature(coral)
        for i in range(core.RESHAPE.space):
            for j in range(core.RESHAPE.time):
                self.assertAlmostEqual(float(coral.temp[i, j]), 300)
        PROCESSES.tme = True


class TestPhotosynthesis(unittest.TestCase):

    def test_initiation(self):
        photosynthesis = Photosynthesis([600, 600], False)
        for i in range(core.RESHAPE.space):
            for j in range(core.RESHAPE.time):
                self.assertEqual(float(photosynthesis.I0[i, j]), 600)
        self.assertFalse(photosynthesis.first_year)
        self.assertEqual(float(photosynthesis.pld), 1)
        self.assertEqual(float(photosynthesis.ptd), 1)
        self.assertEqual(float(photosynthesis.pfd), 1)

    def test_photosynthetic_light_dependency(self):
        photosynthesis = Photosynthesis([600, 600], False)
        coral = Coral(.2, .3, .1, .15, .3)
        coral.initiate_spatial_morphology()
        coral.light = [600, 600]
        photosynthesis.light_dependency(coral, 'qss')
        for i in range(core.RESHAPE.space):
            for j in range(core.RESHAPE.time):
                self.assertAlmostEqual(float(photosynthesis.pld[i, j]), .90727011)

    def test_photosynthetic_flow_dependency(self):
        photosynthesis = Photosynthesis([600, 600], False)
        coral = Coral(.2, .3, .1, .15, .3)
        coral.initiate_spatial_morphology()
        coral.ucm = core.RESHAPE.variable2array([.1, .1])
        photosynthesis.flow_dependency(coral)
        for i in range(core.RESHAPE.space):
            for j in range(core.RESHAPE.time):
                self.assertAlmostEqual(float(photosynthesis.pfd[i, j]), .94485915)
        PROCESSES.pfd = False
        photosynthesis.flow_dependency(coral)
        self.assertEqual(float(photosynthesis.pfd), 1)
        PROCESSES.pfd = True


class TestPopulationStates(unittest.TestCase):
    # TODO: Write tests for the determination of the population states
    pass


class TestCalcification(unittest.TestCase):

    def test_initiation(self):
        calcification = Calcification()
        self.assertEqual(float(calcification.ad), 1)


class TestMorphology(unittest.TestCase):

    def test_initiation(self):
        morphology = Morphology([1, 1], [600, 600])
        for i in range(core.RESHAPE.space):
            self.assertEqual(morphology.calc_sum[i], 1)
            for j in range(core.RESHAPE.time):
                self.assertEqual(morphology.I0[i, j], 600)
        self.assertEqual(morphology.dt_year, 1)
        self.assertEqual(morphology.vol_increase, 0)

        self.assertIsNone(morphology.rf_optimal)
        self.assertIsNone(morphology.rp_optimal)
        self.assertIsNone(morphology.rs_optimal)

    def test_calc_sum_init1(self):
        morphology = Morphology([[1, 1], [1, 1]], [600, 600])
        for i in range(core.RESHAPE.space):
            self.assertEqual(morphology.calc_sum[i], 2)

    def test_optimal_ratios(self):
        morphology = Morphology([1, 1], [600, 600])
        coral = Coral(.2, .3, .1, .15, .3)
        coral.initiate_spatial_morphology()
        coral.light = core.RESHAPE.variable2matrix([600, 600], 'time')
        coral.ucm = core.RESHAPE.variable2array([.1, .1])

        ratios = ('rf', 'rp', 'rs')
        answers = [
            .2,
            .475020813,
            .302412911,
        ]

        for item, ratio in enumerate(ratios):
            setattr(morphology, f'{ratio}_optimal', coral)
            for i in range(core.RESHAPE.space):
                self.assertAlmostEqual(float(getattr(morphology, f'{ratio}_optimal')[i]), answers[item])

    def test_volume_increase(self):
        morphology = Morphology([1, 1], [600, 600])
        coral = Coral(.2, .3, .1, .15, .3)
        coral.initiate_spatial_morphology()
        coral.Bc = DataReshape().variable2matrix(.3, 'time')
        morphology.delta_volume(coral)
        for i in range(core.RESHAPE.space):
            self.assertAlmostEqual(float(morphology.vol_increase[i]), 8.4375e-6)

    def test_morphology_update(self):
        morphology = Morphology([1, 1], [600, 600])
        coral = Coral(.2, .3, .1, .15, .3)
        coral.initiate_spatial_morphology()
        coral.light = core.RESHAPE.variable2matrix([600, 600], 'time')
        coral.ucm = core.RESHAPE.variable2array([.1, .1])
        coral.Bc = core.RESHAPE.variable2matrix([.3, .3], 'time')
        # morphology.delta_volume(coral)
        for ratio in ('rf', 'rp', 'rs'):
            setattr(morphology, f'{ratio}_optimal', coral)
        morphology.update(coral)
        for i in range(core.RESHAPE.space):
            self.assertAlmostEqual(float(coral.rf[i]), 1.498140551)
            self.assertAlmostEqual(float(coral.rp[i]), .499964271)
            self.assertAlmostEqual(float(coral.rs[i]), .666145658)
            self.assertAlmostEqual(float(coral.volume[i]), .005898924)


class TestDislodgement(unittest.TestCase):

    def test_initiation(self):
        dislodgement = Dislodgement()
        self.assertIsNone(dislodgement.dmt)
        self.assertIsNone(dislodgement.csf)
        self.assertIsNone(dislodgement.survival)

    def test_dmt1(self):
        dislodgement = Dislodgement()
        coral = Coral(.2, .3, .1, .15, .3)
        coral.initiate_spatial_morphology()
        coral.um = core.RESHAPE.variable2array([0, 0])
        dislodgement.dislodgement_mechanical_threshold(coral)
        for i in range(core.RESHAPE.space):
            self.assertEqual(dislodgement.dmt[i], 1e20)

    def test_dmt2(self):
        dislodgement = Dislodgement()
        coral = Coral(.2, .3, .1, .15, .3)
        coral.initiate_spatial_morphology()
        coral.um = [.5, .5]
        dislodgement.dislodgement_mechanical_threshold(coral)
        for i in range(core.RESHAPE.space):
            self.assertAlmostEqual(float(dislodgement.dmt[i]), 780.487805, delta=1e-6)

    def test_dmt3(self):
        dislodgement = Dislodgement()
        coral = Coral(.2, .3, .1, .15, .3)
        coral.initiate_spatial_morphology()
        coral.um = [0, .5]
        dislodgement.dislodgement_mechanical_threshold(coral)
        answers = [1e20, 780.487805]
        for i, ans in enumerate(answers):
            self.assertAlmostEqual(float(dislodgement.dmt[i]), ans, delta=1e-6)

    def test_csf1(self):
        dislodgement = Dislodgement()
        coral = Coral(.2, .3, .1, .15, .3)
        coral.initiate_spatial_morphology()
        dislodgement.colony_shape_factor(coral)
        for i in range(core.RESHAPE.space):
            self.assertAlmostEqual(float(dislodgement.csf[i]), 40.1070456591576246)

    def test_csf2(self):
        dislodgement = Dislodgement()
        coral = Coral([.2, 0], [.3, 0], [.1, 0], [.15, 0], [.3, 0])
        dislodgement.colony_shape_factor(coral)
        answers = [40.1070456591576246, 0]
        for i, ans in enumerate(answers):
            self.assertAlmostEqual(float(dislodgement.csf[i]), ans)


# TODO: Recruitment 2x2 tests
# class TestRecruitment(unittest.TestCase):
#
#     def test_spawning_cover1(self):
#         recruitment = Recruitment()
#         coral = Coral(.2, .3, .1, .15, .3)
#         coral.pop_states = np.array(
#             [
#                 [
#                     [1, 0, 0, 0]
#                 ]
#             ]
#         )
#         result = recruitment.spawning(coral, 'P')
#         self.assertEqual(float(result), 0)
#
#     def test_spawning_cover2(self):
#         recruitment = Recruitment()
#         coral = Coral(.2, .3, .1, .15, .3)
#         coral.pop_states = np.array(
#             [
#                 [
#                     [.5, 0, 0, 0]
#                 ]
#             ]
#         )
#         result = recruitment.spawning(coral, 'P')
#         self.assertAlmostEqual(float(result), 2.5e-5)


if __name__ == '__main__':
    unittest.main()
