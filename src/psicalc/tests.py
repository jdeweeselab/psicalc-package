import unittest

import numpy as np
from psicalc import normalized_mutual_info_score as nmi, EPSILON
import psicalc.nmi_util as nu
nu.EPSILON = 1e-9

BASE = 1.0

# python -m unittest src/psicalc/tests.py


class EpsilonTest(unittest.TestCase):
    def test_eq(self):
        tests = [
            (BASE, BASE, True),
            (BASE, BASE + nu.EPSILON / 10, True),
            (BASE, BASE + nu.EPSILON * 10, False),
        ]
        for idx, test in enumerate(tests):
            self.assertEqual(nu.NmiValue(test[0]) == nu.NmiValue(test[1]), test[2], f"{idx}: {test[0]:.20f} == {test[1]:.20f}")

    def test_gt(self):
        tests = [
            (BASE, BASE, False),
            (BASE, BASE + nu.EPSILON / 10, False),
            (BASE, BASE + nu.EPSILON * 10, False),
            (BASE + nu.EPSILON / 10, BASE, False),
            (BASE + nu.EPSILON * 10, BASE, True),
        ]
        for idx, test in enumerate(tests):
            self.assertEqual(nu.NmiValue(test[0]) > nu.NmiValue(test[1]), test[2], f"{idx}: {test[0]} > {test[1]}")

    def test_ge(self):
        tests = [
            (BASE, BASE, True),
            (BASE, BASE + nu.EPSILON / 10, True),
            (BASE, BASE + nu.EPSILON * 10, False),
            (BASE + nu.EPSILON / 10, BASE, True),
            (BASE + nu.EPSILON * 10, BASE, True),
        ]
        for idx, test in enumerate(tests):
            self.assertEqual(nu.NmiValue(test[0]) >= nu.NmiValue(test[1]), test[2], f"{idx}: {test[0]} >= {test[1]}")

    def test_lt(self):
        tests = [
            (BASE, BASE, False),
            (BASE, BASE + nu.EPSILON / 10, False),
            (BASE, BASE + nu.EPSILON * 10, True),
            (BASE + nu.EPSILON / 10, BASE, False),
            (BASE + nu.EPSILON * 10, BASE, False),
        ]
        for idx, test in enumerate(tests):
            self.assertEqual(nu.NmiValue(test[0]) < nu.NmiValue(test[1]), test[2], f"{idx}: {test[0]} < {test[1]}")

    def test_le(self):
        tests = [
            (BASE, BASE, True),
            (BASE, BASE + nu.EPSILON / 10, True),
            (BASE, BASE + nu.EPSILON * 10, True),
            (BASE + nu.EPSILON / 10, BASE, True),
            (BASE + nu.EPSILON * 10, BASE, False),
        ]
        for idx, test in enumerate(tests):
            self.assertEqual(nu.NmiValue(test[0]) <= nu.NmiValue(test[1]), test[2], f"{idx}: {test[0]} <= {test[1]}")

    def test_symmetry(self):
        nu.EPSILON = EPSILON
        tests = [
            # hist (17, 81): 0.7187984457259526
            # hist (81, 17): 0.7187984457259525
            (np.array([11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 0, 11, 11, 11, 0, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 0, 11, 11, 11, 11, 18, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 0, 0]),
             np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0])),
            # hist (128, 127): 1.0
            # hist (127, 128): 0.0
            (np.array([17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 8, 17, 17, 8, 17, 17, 17, 7, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 0]),
             np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 12, 3, 3, 12, 3, 3, 3, 6, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0])),
        ]

        for test in tests:
            x = nmi(test[0], test[1])
            y = nmi(test[1], test[0])
            self.assertEqual(nu.NmiValue(x), nu.NmiValue(y), f"{x} == {y}")


if __name__ == '__main__':
    unittest.main()
