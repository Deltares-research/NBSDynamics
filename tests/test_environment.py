import unittest

from CoralModel_v3.environment import Processes, Constants


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