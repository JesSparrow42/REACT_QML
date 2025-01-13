"""
Simulation SDK Tests
"""

import unittest

import numpy as np
import torch


class SDKTests(unittest.TestCase):
    def test_clifford(self):
        """Tests that clifford_sample gives the correct result"""

        from ptseries.tbi.fixed_random_unitary import FixedRandomUnitary
        from ptseries.tbi.boson_sampler.boson_sampler import clifford_sample
        from tests.references import samples_clifford_ref

        np.random.seed(0)  # the samples of reference have been obtained using random BS with a seed of 0
        x = FixedRandomUnitary.random_unitary_haar(15)
        input_state = np.array([1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0])
        n_samples = 10000

        sample = clifford_sample(x, input_state, n_samples)

        y = np.array(sample)
        y = y.mean(axis=0) / 10

        variation_distance = sum(
            [np.abs(samples_clifford_ref[key] - y[int(key) - 1]) for key in samples_clifford_ref.keys()]
        )

        self.assertLess(variation_distance, 0.05)

    def test_backend_equivalence_inputs(self):
        """Tests that TBISingleLoop and TBIMultiLoop find the same result for the same inputs"""

        from ptseries.tbi.tbi_single_loop import TBISingleLoop
        from ptseries.tbi.tbi_multi_loop import TBIMultiLoop

        # comparison between TBISingleLoop and TBIMultiLoop
        loop = 1
        for input_state in [[1, 1, 1], [1, 1, 1, 1]]:
            n_photons = len(input_state)
            n_samples = 10000
            beam_splitter_angles = np.random.uniform(0, 2 * np.pi, n_photons - 1)

            # Sampling TBI
            tbi_sampler = TBISingleLoop()
            counts_sampler = tbi_sampler.sample(input_state, beam_splitter_angles, n_samples=n_samples)
            probs_sampler = {key: value / n_samples for key, value in counts_sampler.items()}

            # Multi-loops TBI
            tbi_multi_loop = TBIMultiLoop(n_loops=loop)
            counts_multi_loop = tbi_multi_loop.sample(input_state, beam_splitter_angles, n_samples=n_samples)
            probs_multi_loop = {key: value / n_samples for key, value in counts_multi_loop.items()}

            variation_distance = sum(
                [np.abs(probs_sampler[key] - probs_multi_loop.get(key, 0)) for key in probs_sampler.keys()]
            )

            self.assertLess(variation_distance, 0.10)

    def test_backend_equivalence_loss(self):
        """Tests that TBISingleLoop and TBIMultiLoop find the same result for the same loss"""

        from ptseries.tbi.tbi_single_loop import TBISingleLoop
        from ptseries.tbi.tbi_multi_loop import TBIMultiLoop

        loop = 1
        input_state = [1, 1, 0, 1]
        input_loss = 0.5

        n_photons = len(input_state)
        n_samples = 10000
        beam_splitter_angles = np.random.uniform(0, 2 * np.pi, n_photons - 1)

        # Single loop TBI
        tbi_sampler = TBISingleLoop(input_loss=input_loss)
        counts_sampler = tbi_sampler.sample(input_state, beam_splitter_angles, n_samples=n_samples)
        probs_sampler = {key: value / n_samples for key, value in counts_sampler.items()}

        # Multi-loop TBI
        tbi_multi_loop = TBIMultiLoop(n_loops=loop, input_loss=input_loss)
        counts_multi_loop = tbi_multi_loop.sample(input_state, beam_splitter_angles, n_samples=n_samples)
        probs_multi_loop = {key: value / n_samples for key, value in counts_multi_loop.items()}

        variation_distance = sum(
            [np.abs(probs_sampler[key] - probs_multi_loop.get(key, 0)) for key in probs_sampler.keys()]
        )

        self.assertLess(variation_distance, 0.10)

    def test_backend_equivalence_distinguishable(self):
        """Tests that TBISingleLoop and TBIMultiLoop find the same result for the same distinguishable inputs"""

        from ptseries.tbi.tbi_single_loop import TBISingleLoop
        from ptseries.tbi.tbi_multi_loop import TBIMultiLoop

        # comparison between TBISingleLoop and TBIMultiLoop
        loop = 1
        for input_state in [[1, 1, 1], [1, 1, 1, 1]]:
            n_photons = len(input_state)
            n_samples = 10000
            beam_splitter_angles = np.random.uniform(0, 2 * np.pi, n_photons - 1)

            # Sampling TBI
            tbi_sampler = TBISingleLoop(distinguishable=True)
            counts_sampler = tbi_sampler.sample(input_state, beam_splitter_angles, n_samples=n_samples)
            probs_sampler = {key: value / n_samples for key, value in counts_sampler.items()}

            # Multi-loops TBI
            tbi_multi_loop = TBIMultiLoop(n_loops=loop, distinguishable=True)
            counts_multi_loop = tbi_multi_loop.sample(input_state, beam_splitter_angles, n_samples=n_samples)
            probs_multi_loop = {key: value / n_samples for key, value in counts_multi_loop.items()}

            variation_distance = sum(
                [np.abs(probs_sampler[key] - probs_multi_loop.get(key, 0)) for key in probs_sampler.keys()]
            )

            self.assertLess(variation_distance, 0.10)

    def test_backend_equivalence_postselection(self):
        """Tests that TBISingleLoop and TBIMultiLoop find the same result when using postselection."""

        from ptseries.tbi import TBIMultiLoop, TBISingleLoop

        # comparison between TBISingleLoop and TBIMultiLoop
        loop = 1
        for input_state in [[1, 1, 1], [1, 1, 1, 1]]:
            n_photons = len(input_state)
            n_samples = 10000
            beam_splitter_angles = np.random.uniform(0, 2 * np.pi, n_photons - 1)

            # Sampling TBI
            tbi_sampler = TBISingleLoop(bs_loss=0.1, postselected=True)
            counts_sampler = tbi_sampler.sample(input_state, beam_splitter_angles, n_samples=n_samples)
            probs_sampler = {key: value / n_samples for key, value in counts_sampler.items()}

            # Multi-loops TBI
            tbi_multi_loop = TBIMultiLoop(n_loops=loop, bs_loss=0.1, postselected=True)
            counts_multi_loop = tbi_multi_loop.sample(input_state, beam_splitter_angles, n_samples=n_samples)
            probs_multi_loop = {key: value / n_samples for key, value in counts_multi_loop.items()}

            variation_distance = sum(
                [np.abs(probs_sampler[key] - probs_multi_loop.get(key, 0)) for key in probs_sampler.keys()]
            )

            self.assertLess(variation_distance, 0.10)

    def test_multiloop(self):
        """Tests that TBIMultiloop returns the correct distribution"""

        from ptseries.tbi.tbi_multi_loop import TBIMultiLoop
        from tests.references import samples_4modes_2loops_ref

        n_samples = 10000
        n_modes = 4
        n_loops = 2
        np.random.seed(0)  # the samples of reference have been obtained using random BS with a seed of 0
        beam_splitter_angles = np.random.uniform(0, 2 * np.pi, (n_modes - 1) * n_loops)
        input_state = (1, 0, 1, 1)
        tbi_multi_loop = TBIMultiLoop(n_loops=n_loops)
        counts_multi_loop = tbi_multi_loop.sample(input_state, beam_splitter_angles, n_samples=n_samples)
        probs_multi_loop = {key: value / n_samples for key, value in counts_multi_loop.items()}

        variation_distance = sum(
            [
                np.abs(samples_4modes_2loops_ref[key] - probs_multi_loop.get(key, 0))
                for key in samples_4modes_2loops_ref.keys()
            ]
        )

        self.assertLess(variation_distance, 0.10)

    def test_multiloop_with_loss(self):
        """Tests that TBIMultiloop returns the correct distribution"""

        from ptseries.tbi.tbi_multi_loop import TBIMultiLoop
        from tests.references import samples_multi_loop_with_loss_ref

        n_samples = 20000
        n_loops = 2
        input_state = (1, 0, 1, 0)
        np.random.seed(0)
        beam_splitter_angles = np.random.uniform(0, 2 * np.pi, (len(input_state) - 1) * n_loops)
        tbi_multi_loop = TBIMultiLoop(n_loops=n_loops, bs_loss=0.1)
        counts_multi_loop = tbi_multi_loop.sample(input_state, beam_splitter_angles, n_samples=n_samples)
        probs_multi_loop = {key: value / n_samples for key, value in counts_multi_loop.items()}

        variation_distance = sum(
            [
                np.abs(samples_multi_loop_with_loss_ref[key] - probs_multi_loop.get(key, 0))
                for key in samples_multi_loop_with_loss_ref.keys()
            ]
        )

        self.assertLess(variation_distance, 0.05)

    def test_multiloop_with_postselection(self):
        """Tests that TBIMultiLoop with postselection returns the correct distribution."""
        from ptseries.tbi import TBIMultiLoop
        from tests.references import samples_multi_loop_with_postselection_ref

        n_samples = 20000
        n_loops = 2
        input_state = (1, 0, 1, 0)
        np.random.seed(0)
        beam_splitter_angles = np.random.uniform(0, 2 * np.pi, (len(input_state) - 1) * n_loops)
        tbi_multi_loop = TBIMultiLoop(n_loops=n_loops, bs_loss=0.1, postselected=True)
        counts_multi_loop = tbi_multi_loop.sample(input_state, beam_splitter_angles, n_samples=n_samples)
        probs_multi_loop = {key: value / n_samples for key, value in counts_multi_loop.items()}

        variation_distance = sum(
            [
                np.abs(samples_multi_loop_with_postselection_ref[key] - probs_multi_loop.get(key, 0))
                for key in samples_multi_loop_with_postselection_ref.keys()
            ]
        )

        self.assertLess(variation_distance, 0.05)

    def test_single_loop_small_probs(self):
        """Tests on a configuration of angles that has caused numpy/numba issues"""

        from ptseries.tbi import create_tbi

        n_samples = 100000
        beam_splitter_angles = [
            8.5053e-01,
            1.9093e-01,
            -7.1114e-01,
            6.8765e-09,
            9.2746e-01,
            3.2941e00,
            7.1357e-01,
            8.9628e-01,
            5.6369e-01,
            -1.5708e00,
            -3.2166e00,
            2.3169e00,
            3.1277e00,
            2.3054e00,
        ]

        n_modes = len(beam_splitter_angles) + 1
        input_state = [1, 0] * (n_modes // 2) + [1]
        tbi = create_tbi(n_loops=1)

        samples = tbi.sample(input_state, beam_splitter_angles, n_samples=1)
        samples = tbi.sample(input_state, beam_splitter_angles, n_samples=n_samples)

    def test_distinguishable(self):
        """Tests that TBIMultiloop with distinguishable photons returns the correct distribution"""

        from ptseries.tbi.tbi_multi_loop import TBIMultiLoop
        from tests.references import samples_4modes_2loops_distinguishable_ref

        n_samples = 10000
        n_loops = 2
        np.random.seed(0)  # the samples of reference have been obtained using random BS with a seed of 0
        beam_splitter_angles = [np.pi / 3, np.pi / 3, np.pi / 3, np.pi / 6, np.pi / 6, np.pi / 6]
        input_state = (1, 0, 1, 1)
        tbi_distinguishable = TBIMultiLoop(n_loops=n_loops, distinguishable=True)
        counts_distinguishable = tbi_distinguishable.sample(input_state, beam_splitter_angles, n_samples=n_samples)
        probs_distinguishable = {key: value / n_samples for key, value in counts_distinguishable.items()}

        variation_distance = sum(
            [
                np.abs(samples_4modes_2loops_distinguishable_ref[key] - probs_distinguishable.get(key, 0))
                for key in samples_4modes_2loops_distinguishable_ref.keys()
            ]
        )

        self.assertLess(variation_distance, 0.10)

    def test_g2_pseudopnr_singleloop(self):
        """Tests that a single-loop simulator with g2 and pseudo-PNR returns the correct distribution"""

        from ptseries.tbi.tbi_single_loop import TBISingleLoop
        from tests.references import samples_g2_pseudopnr_singleloop_ref as ref

        n_samples = 1000
        tbi = TBISingleLoop(g2=0.5, n_signal_detectors=2, detector_efficiency=0.2, postselected=True)
        samples = tbi.sample(input_state=(1, 1), theta_list=[np.pi / 4], n_samples=n_samples)
        probs = {key: value / n_samples for key, value in samples.items()}

        variation_distance = sum([np.abs(ref[key] - probs.get(key, 0)) for key in ref.keys()])

        self.assertLess(variation_distance, 0.10)

    def test_fixed_random_unitary(self):
        """Tests that FixedRandomUnitary returns the correct distribution"""

        from ptseries.tbi.fixed_random_unitary import FixedRandomUnitary
        from tests.references import samples_haar_ref

        n_samples = 10000
        np.random.seed(0)  # the samples of reference have been obtained using random BS with a seed of 0
        input_state = (1, 0, 1, 1)
        tbi_haar = FixedRandomUnitary()
        counts_haar = tbi_haar.sample(input_state, n_samples=n_samples)
        probs_haar = {key: value / n_samples for key, value in counts_haar.items()}

        variation_distance = sum(
            [np.abs(samples_haar_ref[key] - probs_haar.get(key, 0)) for key in samples_haar_ref.keys()]
        )

        self.assertLess(variation_distance, 0.10)

    def test_multiloop_equivalence(self):
        """Tests that the symbolic and numerical backends for multiloop yield same result"""

        import numpy as np

        from ptseries.tbi.tbi_multi_loop import TBIMultiLoop

        input_state = [1, 1, 1]
        n_loops = 3
        n_samples = 10000
        beam_splitter_angles = np.random.uniform(0, 2 * np.pi, (len(input_state) - 1) * n_loops)

        tbi_multi_loop = TBIMultiLoop(n_loops=n_loops)
        counts_symbolic = tbi_multi_loop.sample(input_state, beam_splitter_angles, n_samples, calculate_symbolic=True)
        probs_symbolic = {key: value / n_samples for key, value in counts_symbolic.items()}
        counts_numeric = tbi_multi_loop.sample(input_state, beam_splitter_angles, n_samples, calculate_symbolic=False)
        probs_numeric = {key: value / n_samples for key, value in counts_numeric.items()}

        variation_distance = sum(
            [np.abs(probs_symbolic[key] - probs_numeric.get(key, 0)) for key in probs_symbolic.keys()]
        )
        self.assertLess(variation_distance, 0.10)

    def test_ptseries_drawing(self):
        """Tests the draw method"""

        import numpy as np

        from ptseries.tbi import create_tbi

        # single-loop
        n_photons = 5
        input_state = [1] * n_photons
        tbi = create_tbi(tbi_type="single-loop")
        tbi.draw(input_state, show_plot=False)
        structure_expected = [[0, 1], [1, 2], [2, 3], [3, 4]]
        self.assertTrue(np.array_equal(np.array(structure_expected), np.array(tbi.structure)))

        # multi-loop
        n_photons = 4
        input_state = [1] * n_photons
        n_loops = 3
        tbi = create_tbi(tbi_type="multi-loop", n_loops=n_loops)
        tbi.draw(input_state, show_plot=False)
        structure_expected = [[0, 1], [1, 2], [2, 3], [0, 1], [1, 2], [2, 3], [0, 1], [1, 2], [2, 3]]
        self.assertTrue(np.array_equal(np.array(structure_expected), np.array(tbi.structure)))

        # fixed-random-unitary
        n_photons = 5
        input_state = [1] * n_photons
        tbi = create_tbi(tbi_type="fixed-random-unitary")
        structure_expected = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
        ]
        tbi.draw(input_state, show_plot=False)
        self.assertTrue(np.array_equal(np.array(structure_expected), np.array(tbi.structure)))

    def test_observables(self):
        """Tests all the observables"""

        from ptseries.models import PTLayer

        beam_splitter_angles = torch.tensor([[np.pi / 4, np.pi / 4]])

        model_avg_photons = PTLayer(
            (1, 1, 1),
            observable="avg-photons",
            n_samples=100,
        )
        outputs_avg_photons = model_avg_photons(beam_splitter_angles)

        model_correlations = PTLayer(
            (1, 1, 1),
            observable="correlations",
            n_samples=100,
        )
        outputs_correlations = model_correlations(beam_splitter_angles)

        model_covariances = PTLayer(
            (1, 1, 1),
            observable="covariances",
            n_samples=100,
        )
        outputs_covariances = model_covariances(beam_splitter_angles)

        model_single_sample = PTLayer(
            (1, 1, 1),
            observable="single-sample",
            n_samples=100,
        )
        outputs_single_sample = model_single_sample(beam_splitter_angles)

    def test_ptlayer_with_tiling(self):
        """Tests if tiling works with PTLayer."""
        from ptseries.models import PTLayer

        ptlayer = PTLayer(
            (1, 0, 0, 0), in_features=4, observable="avg-photons", n_samples=100, tbi_params={"n_loops": 2}, n_tiling=3
        )

        ptlayer.forward(torch.tensor([[0, 0, 0, 0]]))

    def test_ptseries_training(self):
        """Tests that the PTLayer can be trained to route photons from one input to one output"""

        from ptseries.models import PTLayer

        model = PTLayer((1, 0, 0), in_features=0, observable="avg-photons", n_samples=100, tbi_params={"n_loops": 2})

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        objective = torch.tensor([[0.0, 0.0, 1.0]])
        loss = torch.nn.MSELoss()

        for _ in range(1000):
            mse = loss(model(), objective)

            optimizer.zero_grad()
            mse.backward()
            optimizer.step()

        model.eval()
        outputs_final = model()

        self.assertGreater(outputs_final[0, -1].item(), 0.95)

    def test_ptlayer_input_dimension_calculator(self):
        """Tests that the PTLayer input dimension calculator works correctly"""

        from ptseries.models.utils import calculate_n_params

        input_state = (1, 1, 1, 1)

        # These should all return 6
        n_params_1 = calculate_n_params(input_state, tbi_params={"loop_lengths": [1, 2, 3]})
        n_params_2 = calculate_n_params(input_state, tbi_params={"n_loops": 2})
        n_params_3 = calculate_n_params(input_state, n_tiling=2)

        self.assertTrue(n_params_1 == 6 and n_params_2 == 6 and n_params_3 == 6)

    def test_ptlayer_printinfo_method(self):
        """Tests that the PTLayer printinfo method runs successfully"""

        from ptseries.models.pt_layer import PTLayer

        ptlayer = PTLayer((1, 1, 1, 1), in_features=3, observable="avg-photons", tbi_params={"loop_lengths": [1, 2, 3]})
        ptlayer.print_info()

    def test_ptlayer_forward(self):
        from ptseries.models.pt_layer import PTLayer

        input_state = [1, 0]
        n_loops = 1

        pt_layer = PTLayer(
            input_state,
            in_features=0,
            observable="avg-photons",
            tbi_params={"n_loops": n_loops, "distinguishable": True},
            n_samples=10000,
            gradient_mode="parameter-shift",
            gradient_delta=0.3,
        )

        theta_test = 0.2
        pt_layer.set_thetas(torch.tensor([[theta_test]]))

        pt_layer.eval()

        output = pt_layer(x=torch.ones(3, 1))
        output = torch.mean(output, dim=0).detach().numpy()

        forward_theory = [np.cos(theta_test) ** 2, np.sin(theta_test) ** 2]
        dist = sum([np.abs(output[k] - forward_theory[k]) for k in range(output.shape[0])])
        self.assertTrue(dist < 0.03)

    def test_ptlayer_gradient(self):
        from ptseries.models.pt_layer import PTLayer

        input_state = [1, 0]
        n_loops = 1

        pt_layer = PTLayer(
            input_state,
            in_features=0,
            observable="avg-photons",
            tbi_params={"n_loops": n_loops, "distinguishable": True},
            n_samples=10000,
            gradient_mode="parameter-shift",
            gradient_delta=0.3,
        )

        theta_test = 0.2
        pt_layer.set_thetas(torch.tensor([[theta_test]]))

        output = pt_layer(x=torch.ones(3, 1))

        grads = []
        for i in range(output.shape[1]):
            pt_layer.zero_grad()
            loss = torch.mean(output[:, i])
            loss.backward(retain_graph=True)

            grad = pt_layer.theta_trainable.grad.item()
            grads.append(grad)

        grads_theory = [-np.sin(2 * theta_test), np.sin(2 * theta_test)]
        dist = sum([np.abs(grads[k] - grads_theory[k]) for k in range(output.shape[1])])
        self.assertTrue(dist < 0.03)

    def test_ptgenerator_print_info_method(self):
        """Tests that the PTGenerator print_info method runs successfully"""

        from ptseries.models.pt_generator import PTGenerator

        ptgenerator = PTGenerator([1, 1, 1, 1], tbi_params={"loop_lengths": [1, 2, 3]})

        ptgenerator.print_info()

    def test_ptgenerator_backward_method(self):
        """Tests that the PTGenerator backward method runs successfully.

        This tests whether torch.linalg.solve is handled correctly. NB this is not implmented on MPS backend currently
        """
        from ptseries.models.pt_generator import PTGenerator

        ptgenerator = PTGenerator([1, 1, 1, 1], tbi_params={"loop_lengths": [1, 1]})

        samples = ptgenerator.forward(batch_size=1)
        loss = torch.sum(samples)

        ptgenerator.backward(samples, loss.unsqueeze(0))

    # Test Tiled Sample Processing
    def test_tiled_sample_processing(self):
        from ptseries.tbi import create_tbi

        input_state = np.array([1, 0, 0, 0])
        tbi = create_tbi(n_loops=1)
        bs_angles = np.array([np.pi / 4] * (len(input_state) - 1))

        # Test1 : Array Length check
        n_samples = 10000
        n_tiling = 1

        samples = tbi.sample(
            input_state=input_state, theta_list=bs_angles, n_samples=n_samples, output_format="array", n_tiling=n_tiling
        )
        samples_length = len(samples)
        expected_samples_length = n_samples

        self.assertEqual(samples_length, expected_samples_length)

        # Test2 : Array Length check
        n_tiling = np.random.randint(2, 10)
        bs_angles = np.array([np.pi / 4] * ((len(input_state) - 1) * n_tiling))
        samples = tbi.sample(
            input_state=input_state, theta_list=bs_angles, n_samples=n_samples, output_format="array", n_tiling=n_tiling
        )
        samples_length = len(samples)
        expected_samples_length = n_samples

        # Test3 : Shape of tiled array
        samples_shape = (n_samples, len(input_state) * n_tiling)

        self.assertTrue(samples_length == expected_samples_length and samples_shape == samples.shape)

    def test_ptgenerator_tiling(self):
        """Tests tiling functionality with PTGenerator."""
        from ptseries.models.pt_generator import PTGenerator

        batch_size = np.random.randint(5, 10)
        n_tiling = np.random.randint(1, 5)
        input_state = [1, 0, 1, 0, 1, 0, 1, 0]

        ptgenerator = PTGenerator(input_state, tbi_params={"loop_lengths": [1, 2, 4]}, n_tiling=n_tiling)

        samples = ptgenerator.generate(batch_size=batch_size)

        self.assertTrue(samples.shape[0] == batch_size and samples.shape[1] == n_tiling * len(input_state))

    def test_binary_bosonic_solver_training(self):
        """Tests that the binary solver algorithm runs successfully with a QUBO problem."""

        from ptseries.algorithms.binary_solvers import BinaryBosonicSolver

        # Parameters for a "circle" graph, where the minimum is achieved for state (1,1,1,1,1,1)
        M = 6
        Q = np.array(
            [
                [0, -1, 0, 0, 0, -1],
                [-1, 0, 0, 0, -1, 0],
                [0, 0, 0, -1, 0, -1],
                [0, 0, -1, 0, -1, 0],
                [0, -1, 0, -1, 0, 0],
                [-1, 0, -1, 0, 0, 0],
            ]
        )

        def qubo_function(bin_vec):
            return np.dot(bin_vec, np.dot(Q, bin_vec))

        qubo = BinaryBosonicSolver(
            M,
            qubo_function,
            n_samples=50,
        )
        qubo.train(learning_rate=1e-1, updates=5, print_frequency=5)

    def test_binary_bosonic_solver_training_spsa(self):
        """Tests that the BinaryBosonicSolver algorithm runs successfully with a QUBO problem
        using the SPSA optimization algorithm."""

        from ptseries.algorithms.binary_solvers import BinaryBosonicSolver

        # Parameters for a "circle" graph, where the minimum is achieved for state (1,1,1,1,1,1)
        M = 6
        Q = np.array(
            [
                [0, -1, 0, 0, 0, -1],
                [-1, 0, 0, 0, -1, 0],
                [0, 0, 0, -1, 0, -1],
                [0, 0, -1, 0, -1, 0],
                [0, -1, 0, -1, 0, 0],
                [-1, 0, -1, 0, 0, 0],
            ]
        )

        def qubo_function(bin_vec):
            return np.dot(bin_vec, np.dot(Q, bin_vec))

        qubo = BinaryBosonicSolver(
            M,
            qubo_function,
            n_samples=50,
            gradient_mode="spsa",
            spsa_params={"spsa_resamplings": 2, "optimizer_quantum": "Adam"},
        )
        qubo.train(learning_rate=1e-1, updates=5, print_frequency=5)

    def test_binary_bosonic_solver_training_finite_difference(self):
        """Tests that the BinaryBosonicSolver algorithm runs successfully with a QUBO problem
        using the SPSA optimization algorithm."""

        from ptseries.algorithms.binary_solvers import BinaryBosonicSolver

        # Parameters for a "circle" graph, where the minimum is achieved for state (1,1,1,1,1,1)
        M = 6
        Q = np.array(
            [
                [0, -1, 0, 0, 0, -1],
                [-1, 0, 0, 0, -1, 0],
                [0, 0, 0, -1, 0, -1],
                [0, 0, -1, 0, -1, 0],
                [0, -1, 0, -1, 0, 0],
                [-1, 0, -1, 0, 0, 0],
            ]
        )

        def qubo_function(bin_vec):
            return np.dot(bin_vec, np.dot(Q, bin_vec))

        qubo = BinaryBosonicSolver(
            M,
            qubo_function,
            n_samples=50,
            gradient_mode="finite-difference",
        )
        qubo.train(learning_rate=1e-1, updates=5, print_frequency=5)

    def test_pt_gen_distinguishable_sampler(self):
        from ptseries.models import PTGenerator
        from ptseries.tbi.tbi import create_tbi

        n_samples = 10000
        input_state = [1, 0, 1, 0]
        distinguishable = True
        detector_efficiency = 0.85
        input_loss = 0.18
        n_loops = 1
        tbi_params = {
            "n_loops": n_loops,
            "distinguishable": distinguishable,
            "detector_efficiency": detector_efficiency,
            "input_loss": input_loss,
        }
        latent_space = PTGenerator(input_state=input_state, tbi_params=tbi_params)

        samples1 = latent_space(n_samples)
        samples_dict1 = {}
        for state in samples1.numpy():
            state_tuple = tuple(map(int, state))
            samples_dict1[state_tuple] = samples_dict1.get(state_tuple, 0) + 1

        bs_angles = latent_space.theta_trainable.detach().numpy()
        tbi = create_tbi(
            distinguishable=distinguishable,
            detector_efficiency=detector_efficiency,
            input_loss=input_loss,
            n_loops=n_loops,
        )
        samples2_dict = tbi.sample(input_state, bs_angles, n_samples=n_samples)

        error = 0
        for state, value in samples2_dict.items():
            error += np.abs(value - samples_dict1.get(state, 0)) / n_samples

        self.assertLess(error, 0.08)

    def test_torch_device(self):
        """
        Tests that torch tensors are sent to devices as expected and can run basic forward and backward"""
        from ptseries.algorithms.gans.gan import WGANGP
        from tutorial_notebooks.utils.mnist_models import Generator, Critic

        # Create dummy network
        gen = Generator(latent_dim=4)
        crit = Critic()
        latent = None
        gan = WGANGP(latent, gen, crit)
        input_state = torch.tensor(
            [[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]], device=gan.device
        )  # dummy latent vecs

        # forward
        out = gan.generator(input_state)
        loss = torch.mean(out)

        # backward
        loss.backward()

    def test_postselected_sampling_single_loop(self):
        """
        Tests that the correct number of samples is returned and that each sample has the same photon number as the
        input state for a single-loop TBI.
        """
        from ptseries.tbi.tbi import create_tbi

        input_state = [1, 0] * 6
        n_photons = sum(input_state)
        n_samples = 100
        n_loops = 1

        tbi_single = create_tbi(
            n_loops=n_loops, bs_loss=0.1, input_loss=0.1, detector_efficiency=0.9, postselected=True
        )
        beam_splitter_angles = np.random.uniform(0, 2 * np.pi, (len(input_state) - 1) * n_loops)

        samples = tbi_single.sample(input_state, beam_splitter_angles, n_samples)

        # check that all samples have same photon number as input
        n_samples_returned = 0
        for sample, occur in samples.items():
            n_samples_returned += occur
            if sum(sample) != n_photons:
                raise ValueError("Output sample photon number does not equal input photon number.")

        if n_samples_returned != n_samples:
            raise ValueError("Incorrect number of samples returned.")


if __name__ == "__main__":
    unittest.main()
