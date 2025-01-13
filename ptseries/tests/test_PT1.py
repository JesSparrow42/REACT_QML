"""
PT-1 SDK Tests
"""

import unittest


class PT1Tests(unittest.TestCase):
    def test_photon_routing(self):
        """Tests that single photons are correctly routed"""
        import numpy as np
        from ptseries.tbi import create_tbi

        # create a two mode tbi with a 50:50 BS
        time_bin_interferometer = create_tbi(tbi_type="PT-1")

        samples = time_bin_interferometer.sample(input_state=(1, 0), theta_list=[np.pi / 4], n_samples=200)

        splitting_ratio = samples[(1, 0)] / samples[(0, 1)]

        self.assertTrue(0.7 < splitting_ratio < 1.3)

        samples = time_bin_interferometer.sample(input_state=(1, 0), theta_list=[np.pi / 2], n_samples=200)

        splitting_ratio = samples[(0, 1)] / samples.get((1, 0), 1)

        self.assertTrue(splitting_ratio > 0.7)


if __name__ == "__main__":
    unittest.main()
