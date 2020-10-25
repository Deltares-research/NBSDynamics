import unittest

import numpy

from CoralModel_v3.utils import SpaceTime, DataReshape, Processes, Constants


class TestSpaceTime(unittest.TestCase):

    def test_default(self):
        spacetime = SpaceTime().spacetime
        self.assertEqual(len(spacetime), 2)
        self.assertIsInstance(spacetime, tuple)

    def test_default_input(self):
        spacetime = SpaceTime(None).spacetime
        self.assertEqual(len(spacetime), 2)
        self.assertIsInstance(spacetime, tuple)

    def test_global_raise_type_error(self):
        with self.assertRaises(TypeError):
            SpaceTime(int(1))
        with self.assertRaises(TypeError):
            SpaceTime(float(1))
        with self.assertRaises(TypeError):
            SpaceTime(str(1))

    def test_global_not_raise_type_error(self):
        SpaceTime((1, 1))
        SpaceTime([1, 1])

    def test_size_error(self):
        with self.assertRaises(ValueError):
            SpaceTime((1,))
        with self.assertRaises(ValueError):
            SpaceTime((1, 1, 1))

    def test_local_raise_type_error(self):
        with self.assertRaises(TypeError):
            SpaceTime((float(1), 1))

    def test_return_type(self):
        self.assertIsInstance(SpaceTime((1, 1)).spacetime, tuple)
        self.assertIsInstance(SpaceTime([1, 1]).spacetime, tuple)


class TestDataReshape(unittest.TestCase):

    def test_default_spacetime(self):
        reshape = DataReshape()
        spacetime = SpaceTime().spacetime
        self.assertIsInstance(reshape.spacetime, tuple)
        self.assertEqual(reshape.spacetime, spacetime)

    def test_default_space(self):
        reshape = DataReshape()
        spacetime = SpaceTime().spacetime
        self.assertIsInstance(reshape.space, int)
        self.assertEqual(reshape.space, spacetime[0])

    def test_default_time(self):
        reshape = DataReshape()
        spacetime = SpaceTime().spacetime
        self.assertIsInstance(reshape.time, int)
        self.assertEqual(reshape.time, spacetime[1])

    def test_set_spacetime_raise_type_error(self):
        with self.assertRaises(TypeError):
            DataReshape(int(1))
        with self.assertRaises(TypeError):
            DataReshape(float(1))
        with self.assertRaises(TypeError):
            DataReshape(str(1))

    def test_set_spacetime_not_raise_error(self):
        DataReshape((1, 1))
        DataReshape([1, 1])

    def test_variable2array(self):
        self.assertIsInstance(DataReshape.variable2array(float(1)), numpy.ndarray)
        self.assertIsInstance(DataReshape.variable2array(int(1)), numpy.ndarray)
        with self.assertRaises(NotImplementedError):
            DataReshape.variable2array(str(1))
        self.assertIsInstance(DataReshape.variable2array((1, 1)), numpy.ndarray)
        self.assertIsInstance(DataReshape.variable2array([1, 1]), numpy.ndarray)

    def test_variable2matrix_shape_space(self):
        reshape = DataReshape((4, 5))
        var = [0, 1, 2, 4]
        matrix = reshape.variable2matrix(var, 'space')
        self.assertEqual(matrix.shape, (4, 5))

    def test_variable2matrix_shape_time(self):
        reshape = DataReshape((4, 5))
        var = [0, 1, 2, 4, 8]
        matrix = reshape.variable2matrix(var, 'time')
        self.assertEqual(matrix.shape, (4, 5))

    def test_variable2matrix_value_space(self):
        reshape = DataReshape((4, 5))
        var = [0, 1, 2, 4]
        matrix = reshape.variable2matrix(var, 'space')
        for i, row in enumerate(matrix):
            for col in row:
                self.assertEqual(col, var[i])

    def test_variable2matrix_value_time(self):
        reshape = DataReshape((4, 5))
        var = [0, 1, 2, 4, 8]
        matrix = reshape.variable2matrix(var, 'time')
        for row in matrix:
            for i, col in enumerate(row):
                self.assertEqual(col, var[i])

    def test_raise_error_space(self):
        reshape = DataReshape((4, 5))
        var = [0, 1, 2, 4, 8]
        with self.assertRaises(ValueError):
            reshape.variable2matrix(var, 'space')

    def test_raise_error_time(self):
        reshape = DataReshape((4, 5))
        var = [0, 1, 2, 4]
        with self.assertRaises(ValueError):
            reshape.variable2matrix(var, 'time')

    def test_matrix2array_space_last(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'space', None)
        answer = [8, 8, 8, 8]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)

    def test_matrix2array_space_mean(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'space', 'mean')
        answer = [3, 3, 3, 3]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)

    def test_matrix2array_space_max(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'space', 'max')
        answer = [8, 8, 8, 8]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)

    def test_matrix2array_space_min(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'space', 'min')
        answer = [0, 0, 0, 0]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)

    def test_matrix2array_space_sum(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'space', 'sum')
        answer = [15, 15, 15, 15]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)

    def test_matrix2array_time_last(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'time', None)
        answer = [0, 1, 2, 4, 8]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)

    def test_matrix2array_time_mean(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'time', 'mean')
        answer = [0, 1, 2, 4, 8]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)

    def test_matrix2array_time_max(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'time', 'max')
        answer = [0, 1, 2, 4, 8]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)

    def test_matrix2array_time_min(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'time', 'min')
        answer = [0, 1, 2, 4, 8]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)

    def test_matrix2array_time_sum(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'time', 'sum')
        answer = [0, 4, 8, 16, 32]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)


class TestProcesses(unittest.TestCase):

    def test_default(self):
        processes = Processes()
        self.assertTrue(processes.fme)
        self.assertTrue(processes.tme)
        self.assertTrue(processes.pfd)

    def test_tme_false(self):
        processes = Processes(tme=False)
        self.assertTrue(processes.fme)
        self.assertFalse(processes.tme)
        self.assertTrue(processes.pfd)

    def test_fme_false(self):
        processes = Processes(fme=False)
        self.assertFalse(processes.fme)
        self.assertFalse(processes.tme)
        self.assertTrue(processes.pfd)

    def test_pfd_false(self):
        processes = Processes(pfd=False)
        self.assertFalse(processes.fme)
        self.assertFalse(processes.tme)
        self.assertFalse(processes.pfd)


class TestConstants(unittest.TestCase):

    def test_default_function(self):
        constants = Constants(Processes(), turbulence_coef=.2)
        self.assertEqual(constants.Cs, .2)

    def test_default_lme(self):
        constants = Constants(Processes())
        self.assertEqual(constants.Kd0, .1)
        self.assertAlmostEqual(constants.theta_max, 1.5707963267948966)

    def test_default_fme(self):
        constants = Constants(Processes())
        self.assertEqual(constants.Cs, .17)
        self.assertEqual(constants.Cm, 1.7)
        self.assertEqual(constants.Cf, .01)
        self.assertEqual(constants.nu, 1e-6)
        self.assertEqual(constants.alpha, 1e-7)
        self.assertEqual(constants.psi, 2)
        self.assertEqual(constants.wcAngle, 0)
        self.assertEqual(constants.rd, 500)
        self.assertEqual(constants.numericTheta, .5)
        self.assertEqual(constants.err, 1e-3)
        self.assertEqual(constants.maxiter_k, 1e5)
        self.assertEqual(constants.maxiter_aw, 1e5)

    def test_default_tme(self):
        constants = Constants(Processes())
        self.assertEqual(constants.K0, 80)
        self.assertEqual(constants.ap, .4)
        self.assertEqual(constants.k, .6089)

    def test_default_pld(self):
        constants = Constants(Processes())
        self.assertEqual(constants.iota, .6)
        self.assertEqual(constants.ik_max, 372.32)
        self.assertEqual(constants.pm_max, 1)
        self.assertEqual(constants.betaI, .34)
        self.assertEqual(constants.betaP, .09)

    def test_default_ptd(self):
        constants = Constants(Processes())
        self.assertEqual(constants.Ea, 6e4)
        self.assertEqual(constants.R, 8.31446261815324)
        self.assertEqual(constants.k_var, 2.45)
        self.assertEqual(constants.nn, 60)

    def test_default_pfd(self):
        constants = Constants(Processes())
        self.assertEqual(constants.pfd_min, .68886964)
        self.assertEqual(constants.ucr, .17162374)

        constants = Constants(Processes(fme=False))
        self.assertEqual(constants.pfd_min, .68886964)
        self.assertEqual(constants.ucr, .5173)

    def test_default_pd(self):
        constants = Constants(Processes())
        self.assertEqual(constants.r_growth, .002)
        self.assertEqual(constants.r_recovery, .2)
        self.assertEqual(constants.r_mortality, .04)
        self.assertEqual(constants.r_bleaching, 8)

    def test_default_c(self):
        constants = Constants(Processes())
        self.assertEqual(constants.gC, .5)
        self.assertEqual(constants.omegaA0, 5)
        self.assertEqual(constants.omega0, .14587415)
        self.assertEqual(constants.kappaA, .66236107)

    def test_default_md(self):
        constants = Constants(Processes())
        self.assertEqual(constants.prop_form, .1)
        self.assertEqual(constants.prop_plate, .5)
        self.assertEqual(constants.prop_plate_flow, .1)
        self.assertAlmostEqual(constants.prop_space, .35355339059327373)
        self.assertEqual(constants.prop_space_light, .1)
        self.assertEqual(constants.prop_space_flow, .1)
        self.assertEqual(constants.u0, .2)
        self.assertEqual(constants.rho_c, 1600)

    def test_default_dc(self):
        constants = Constants(Processes())
        self.assertEqual(constants.sigma_t, 2e5)
        self.assertEqual(constants.Cd, 1)
        self.assertEqual(constants.rho_w, 1025)

    def test_default_cr(self):
        constants = Constants(Processes())
        self.assertEqual(constants.no_larvae, 1e6)
        self.assertEqual(constants.prob_settle, 1e-4)
        self.assertEqual(constants.d_larvae, 1e-3)


if __name__ == '__main__':
    unittest.main()
