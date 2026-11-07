import unittest

import numpy as np

from semwave.anisotropy import phase_velocities, thomsen_stiffness
from semwave.chaboche import Chaboche, elastic_increment
from semwave.random_field import shinozuka_deodatis_2d


class RandomFieldTests(unittest.TestCase):
    def test_seeded_realisation_is_reproducible(self):
        x1 = np.linspace(0.0, 1000.0, 31)
        x2 = np.linspace(0.0, 500.0, 17)
        kwargs = dict(sigma=0.15, b1=250.0, b2=100.0, n_wave=(8, 6))
        first = shinozuka_deodatis_2d(
            x1, x2, rng=np.random.default_rng(12), **kwargs
        )
        second = shinozuka_deodatis_2d(
            x1, x2, rng=np.random.default_rng(12), **kwargs
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (x2.size, x1.size))
        self.assertTrue(np.isfinite(first).all())


class AnisotropyTests(unittest.TestCase):
    def test_zero_thomsen_parameters_recover_isotropy(self):
        lam, mu, rho = 4.0, 2.0, 2.0
        stiffness = thomsen_stiffness(lam, mu, epsilon=0.0, delta=0.0)
        self.assertEqual(stiffness, (lam + 2.0 * mu, lam,
                                     lam + 2.0 * mu, mu))
        angle = np.linspace(0.0, np.pi, 41)
        qp, qsv = phase_velocities(*stiffness, rho, angle)
        np.testing.assert_allclose(qp, np.sqrt((lam + 2.0 * mu) / rho))
        np.testing.assert_allclose(qsv, np.sqrt(mu / rho))


class ChabocheTests(unittest.TestCase):
    def test_elastic_step_matches_hooke_increment(self):
        deps = np.array([[1.0e-6, -0.25e-6, 0.0, 0.5e-6]])
        lam = np.array([4.0e9])
        mu = np.array([2.0e9])
        sig = np.zeros((1, 4))
        backstress = np.zeros((1, 4))
        isotropic = np.zeros(1)
        pacc = np.zeros(1)
        law = Chaboche(sigma_y=np.inf, C_kin=0.0)

        expected = elastic_increment(deps, lam, mu)
        increment, yielded = law.update(
            sig, backstress, isotropic, pacc, deps, lam, mu,
            dt=1.0e-3, active=np.ones(1, dtype=bool)
        )

        self.assertEqual(yielded, 0)
        np.testing.assert_allclose(increment, expected)
        np.testing.assert_allclose(sig, expected)
        np.testing.assert_array_equal(pacc, 0.0)

    def test_overstress_produces_finite_plastic_flow(self):
        deps = np.array([[2.0e-3, -1.0e-3, 0.0, 0.5e-3]])
        lam = np.array([4.0e9])
        mu = np.array([2.0e9])
        sig = np.zeros((1, 4))
        backstress = np.zeros((1, 4))
        isotropic = np.zeros(1)
        pacc = np.zeros(1)
        law = Chaboche(
            sigma_y=1.0e6, C_kin=5.0e6, Q_iso=2.0e5,
            K_visc=1.0e5, n_visc=3.0
        )

        _, yielded = law.update(
            sig, backstress, isotropic, pacc, deps, lam, mu,
            dt=1.0e-3, active=np.ones(1, dtype=bool)
        )

        self.assertEqual(yielded, 1)
        self.assertGreater(pacc[0], 0.0)
        self.assertTrue(np.isfinite(sig).all())
        self.assertTrue(np.isfinite(backstress).all())


if __name__ == "__main__":
    unittest.main()
