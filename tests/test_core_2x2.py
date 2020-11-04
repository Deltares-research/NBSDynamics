import unittest

import numpy as np

from CoralModel_v3 import core
from CoralModel_v3.core import CONSTANTS, Coral, Light, Flow, Temperature, PROCESSES, Photosynthesis, \
    Calcification, Morphology, Dislodgement, Recruitment
from CoralModel_v3.utils import DataReshape

# TODO: Rewrite tests core for "spacetime = (2, 2)". Modulating RESHAPE.spacetime in this file results in problems,
#  as for every test the spacetime has to be redefined. Therefore, group them to test the core for non-floats
#  (i.e. matrices) as well.


core.RESHAPE = DataReshape((2, 2))


class TestCoral(unittest.TestCase):

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

#     def test_vegetation_density(self):
#         coral = Coral(.2, .3, .1, .15, .3)
#         self.assertAlmostEqual(coral.as_vegetation_density, 3.3333333333333335)
#
#     def test_coral_cover(self):
#         coral = Coral(.2, .3, .1, .15, .3)
#         self.assertEqual(coral.cover, 1)
#         coral = Coral([.2, .2], [.3, .3], [.1, .1], [.15, .15], [.3, .3])
#         self.assertEqual(coral.cover[0], 1)
#         self.assertEqual(coral.cover[1], 1)
#
#
# class TestLight(unittest.TestCase):
#
#     def test_initiation(self):
#         light = Light(600, .1, 5)
#         self.assertEqual(light.I0, 600)
#         self.assertEqual(light.Kd, .1)
#         self.assertEqual(light.h, 5)
#
#     def test_representative_light(self):
#         light = Light(600, .1, 5)
#         # base light
#         coral = Coral(.2, .3, .1, .15, .3)
#         light.rep_light(coral)
#         answer = 217.0490558
#         self.assertAlmostEqual(float(coral.light), answer)
#         # no base light
#         coral = Coral(.4, .3, .2, .15, .3)
#         light.rep_light(coral)
#         answer = 253.8318634
#         self.assertAlmostEqual(float(coral.light), answer)
#
#     def test_coral_biomass(self):
#         light = Light(600, .1, 5)
#         # base light
#         coral = Coral(.2, .3, .1, .15, .3)
#         light.biomass(coral)
#         answer = .14287642
#         self.assertAlmostEqual(float(coral.Bc), answer)
#         # no base light
#         coral = Coral(.4, .3, .2, .15, .3)
#         light.biomass(coral)
#         answer = .314159265
#         self.assertAlmostEqual(float(coral.Bc), answer)
#
#     def test_base_light(self):
#         light = Light(600, .1, 5)
#         # base light
#         coral = Coral(.2, .3, .1, .15, .3)
#         result = light.base_light(coral)
#         answer = .05478977
#         self.assertAlmostEqual(float(result), answer)
#         # no base light
#         coral = Coral(.4, .3, .2, .15, .3)
#         result = light.base_light(coral)
#         answer = 0
#         self.assertAlmostEqual(float(result), answer)
#
#     def test_side_correction(self):
#         light = Light(600, .1, 5)
#         coral = Coral(.2, .3, .1, .15, .3)
#         max_thetas = np.linspace(0, np.pi)
#         for theta in max_thetas:
#             CONSTANTS.theta_max = theta
#             result = light.side_correction(coral)
#             self.assertLess(result, 1)
#
#
# class TestFlow(unittest.TestCase):
#
#     def test_initiation(self):
#         flow = Flow(.1, .1, 5, 4)
#         self.assertEqual(flow.uc[0], .1)
#         self.assertEqual(flow.uw[0], .1)
#         self.assertEqual(flow.h, 5)
#         self.assertEqual(flow.Tp[0], 4)
#
#     def test_wave_attenuation(self):
#         # input array
#         diameter = [.1, .2, .4]     # [m]
#         height = .3                 # [m]
#         distance = [.3, .4, .6]     # [m]
#         velocity = .05              # [m s-1]
#         period = 4                  # [s]
#         depth = .75                 # [m]
#
#         # answers
#         answer = [
#             .73539733818684030,
#             .47628599416211803,
#             .20277038395777466,
#         ]
#
#         for i in range(3):
#             wac = Flow.wave_attenuation(
#                 diameter=diameter[i],
#                 height=height,
#                 distance=distance[i],
#                 velocity=velocity,
#                 period=period,
#                 depth=depth,
#                 wac_type='wave',
#             )
#             self.assertAlmostEqual(float(wac), answer[i], delta=.1)
#
#
# class TestTemperature(unittest.TestCase):
#
#     def test_initiation(self):
#         temperature = Temperature(300)
#         self.assertEqual(temperature.T, 300)
#
#     def test_coral_temperature(self):
#         temperature = Temperature(300)
#         coral = Coral(.2, .3, .1, .15, .3)
#         coral.delta_t = .001
#         coral.light = 600
#         temperature.coral_temperature(coral)
#         self.assertAlmostEqual(float(coral.temp), 300.00492692)
#
#     def test_no_tme(self):
#         PROCESSES.tme = False
#         temperature = Temperature(300)
#         coral = Coral(.2, .3, .1, .15, .3)
#         coral.delta_t = .001
#         coral.light = 600
#         temperature.coral_temperature(coral)
#         self.assertAlmostEqual(float(coral.temp), 300)
#         PROCESSES.tme = True
#
#
# class TestPhotosynthesis(unittest.TestCase):
#
#     def test_initiation(self):
#         photosynthesis = Photosynthesis(600, False)
#         self.assertEqual(float(photosynthesis.I0), 600)
#         self.assertFalse(photosynthesis.first_year)
#         self.assertEqual(float(photosynthesis.pld), 1)
#         self.assertEqual(float(photosynthesis.ptd), 1)
#         self.assertEqual(float(photosynthesis.pfd), 1)
#
#     def test_photosynthetic_light_dependency(self):
#         photosynthesis = Photosynthesis(600, False)
#         coral = Coral(.2, .3, .1, .15, .3)
#         coral.light = 600
#         photosynthesis.light_dependency(coral, 'qss')
#         self.assertAlmostEqual(float(photosynthesis.pld), .90727011)
#
#     def test_photosynthetic_flow_dependency(self):
#         photosynthesis = Photosynthesis(600, False)
#         coral = Coral(.2, .3, .1, .15, .3)
#         coral.ucm = .1
#         photosynthesis.flow_dependency(coral)
#         self.assertAlmostEqual(float(photosynthesis.pfd), .94485915)
#         PROCESSES.pfd = False
#         photosynthesis.flow_dependency(coral)
#         self.assertEqual(float(photosynthesis.pfd), 1)
#         PROCESSES.pfd = True
#
#
# class TestPopulationStates(unittest.TestCase):
#     # TODO: Write tests for the determination of the population states
#     pass
#
#
# class TestCalcification(unittest.TestCase):
#
#     def test_initiation(self):
#         calcification = Calcification()
#         self.assertEqual(float(calcification.ad), 1)
#
#     def test_calcification_rate(self):
#         calcification = Calcification()
#         coral = Coral(.2, .3, .1, .15, .3)
#         coral.pop_states = np.array([[[[1]]]])
#         coral.photo_rate = 1
#         omegas = np.linspace(1, 5, 4)
#         answer = [
#             .28161333,
#             .38378897,
#             .42082994,
#             .43996532
#         ]
#         for i, omega in enumerate(omegas):
#             calcification.calcification_rate(coral, omega)
#             self.assertAlmostEqual(float(coral.calc), answer[i])
#
#
# class TestMorphology(unittest.TestCase):
#
#     def test_initiation(self):
#         morphology = Morphology(1, 600)
#         self.assertEqual(morphology.calc_sum, 1)
#         self.assertEqual(morphology.I0, 600)
#         self.assertEqual(morphology.dt_year, 1)
#         self.assertEqual(morphology.vol_increase, 0)
#
#         self.assertIsNone(morphology.rf_optimal)
#         self.assertIsNone(morphology.rp_optimal)
#         self.assertIsNone(morphology.rs_optimal)
#
#     def test_calc_sum_init1(self):
#         morphology = Morphology([1, 1], 600)
#         answer = [1, 1]
#         for i, val in enumerate(answer):
#             self.assertEqual(morphology.calc_sum[i], val)
#
#     def test_optimal_ratios(self):
#         morphology = Morphology(1, DataReshape().variable2matrix(600, 'time'))
#         coral = Coral(.2, .3, .1, .15, .3)
#         coral.light = DataReshape().variable2matrix(600, 'time')
#         coral.ucm = DataReshape().variable2array(.1)
#
#         ratios = ('rf', 'rp', 'rs')
#         answers = [
#             .2,
#             .475020813,
#             .302412911,
#         ]
#
#         for i, ratio in enumerate(ratios):
#             setattr(morphology, f'{ratio}_optimal', coral)
#             self.assertAlmostEqual(float(getattr(morphology, f'{ratio}_optimal')), answers[i])
#
#     def test_volume_increase(self):
#         morphology = Morphology(1, DataReshape().variable2matrix(600, 'time'))
#         coral = Coral(.2, .3, .1, .15, .3)
#         coral.Bc = DataReshape().variable2matrix(.3, 'time')
#         morphology.delta_volume(coral)
#         self.assertAlmostEqual(float(morphology.vol_increase), 8.4375e-6)
#
#     def test_morphology_update(self):
#         morphology = Morphology(1, DataReshape().variable2matrix(600, 'time'))
#         coral = Coral(.2, .3, .1, .15, .3)
#         coral.light = DataReshape().variable2matrix(600, 'time')
#         coral.ucm = DataReshape().variable2array(.1)
#         coral.Bc = DataReshape().variable2matrix(.3, 'time')
#         # morphology.delta_volume(coral)
#         for ratio in ('rf', 'rp', 'rs'):
#             setattr(morphology, f'{ratio}_optimal', coral)
#         morphology.update(coral)
#         self.assertAlmostEqual(float(coral.rf), 1.498140551)
#         self.assertAlmostEqual(float(coral.rp), .499964271)
#         self.assertAlmostEqual(float(coral.rs), .666145658)
#         self.assertAlmostEqual(float(coral.volume), .005898924)
#
#
# class TestDislodgement(unittest.TestCase):
#
#     def test_initiation(self):
#         dislodgement = Dislodgement()
#         self.assertIsNone(dislodgement.dmt)
#         self.assertIsNone(dislodgement.csf)
#         self.assertIsNone(dislodgement.survival)
#
#     def test_dmt1(self):
#         dislodgement = Dislodgement()
#         coral = Coral(.2, .3, .1, .15, .3)
#         coral.um = 0
#         dislodgement.dislodgement_mechanical_threshold(coral)
#         self.assertEqual(dislodgement.dmt, 1e20)
#
#     def test_dmt2(self):
#         dislodgement = Dislodgement()
#         coral = Coral(.2, .3, .1, .15, .3)
#         coral.um = .5
#         dislodgement.dislodgement_mechanical_threshold(coral)
#         self.assertAlmostEqual(float(dislodgement.dmt), 780.487805, delta=1e-6)
#
#     def test_dmt3(self):
#         dislodgement = Dislodgement()
#         coral = Coral([.2, .2], [.3, .3], [.1, .1], [.15, .15], [.3, .3])
#         coral.um = [0, .5]
#         dislodgement.dislodgement_mechanical_threshold(coral)
#         answers = [1e20, 780.487805]
#         for i, ans in enumerate(answers):
#             self.assertAlmostEqual(float(dislodgement.dmt[i]), ans, delta=1e-6)
#
#     def test_csf1(self):
#         dislodgement = Dislodgement()
#         coral = Coral(.2, .3, .1, .15, .3)
#         dislodgement.colony_shape_factor(coral)
#         self.assertAlmostEqual(float(dislodgement.csf), 40.1070456591576246)
#
#     def test_csf2(self):
#         dislodgement = Dislodgement()
#         coral = Coral([.2, 0], [.3, 0], [.1, 0], [.15, 0], [.3, 0])
#         dislodgement.colony_shape_factor(coral)
#         answers = [40.1070456591576246, 0]
#         for i, ans in enumerate(answers):
#             self.assertAlmostEqual(float(dislodgement.csf[i]), ans)
#
#
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
