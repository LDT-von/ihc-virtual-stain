"""Focused structural checks for the DAPI-only prototype marker candidate."""

import unittest

import torch

from src.models.prototype_marker import PrototypeMarkerNet
from src.train_marker_context import build_reconstruction_model, predict


class PrototypeMarkerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    @staticmethod
    def make_model():
        return PrototypeMarkerNet(
            width=4,
            markers=4,
            context=True,
            num_shared_prototypes=3,
            num_task_prototypes=2,
            temperature=0.25,
        )

    def test_odd_rectangular_input_preserves_shape_and_output_range(self):
        model = self.make_model().eval()
        x = torch.rand(2, 1, 17, 23)
        with torch.no_grad():
            output = model(x)
        self.assertEqual(output.shape, (2, 4, 17, 23))
        self.assertTrue(torch.isfinite(output).all().item())
        self.assertTrue(((output >= 0) & (output <= 1)).all().item())

    def test_all_marker_decoders_and_prototype_banks_receive_reconstruction_gradient(self):
        model = self.make_model()
        output = model(torch.rand(1, 1, 17, 23))
        target = torch.rand_like(output)
        (output - target).square().mean().backward()

        self.assertEqual(len(model.decoders), 4)
        for marker_index, decoder in enumerate(model.decoders):
            with self.subTest(marker=marker_index):
                gradient = decoder.head.weight.grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all().item())
                self.assertGreater(gradient.abs().sum().item(), 0)

        shared_gradient = model.router.shared.grad
        private_gradient = model.router.private.grad
        self.assertIsNotNone(shared_gradient)
        self.assertIsNotNone(private_gradient)
        self.assertEqual(private_gradient.shape[:2], (4, 2))
        self.assertTrue(torch.isfinite(shared_gradient).all().item())
        self.assertTrue(torch.isfinite(private_gradient).all().item())
        self.assertGreater(shared_gradient.abs().sum().item(), 0)
        for marker_index in range(4):
            with self.subTest(prototype_marker=marker_index):
                self.assertGreater(private_gradient[marker_index].abs().sum().item(), 0)

    def test_diversity_penalty_is_finite_differentiable_and_reaches_both_banks(self):
        model = self.make_model()
        penalty = model.prototype_diversity_loss()
        self.assertEqual(penalty.ndim, 0)
        self.assertTrue(penalty.requires_grad)
        self.assertTrue(torch.isfinite(penalty).item())
        self.assertGreaterEqual(penalty.item(), 0)

        penalty.backward()
        for bank in (model.router.shared, model.router.private):
            self.assertIsNotNone(bank.grad)
            self.assertTrue(torch.isfinite(bank.grad).all().item())
            self.assertGreater(bank.grad.abs().sum().item(), 0)

    def test_invalid_constructor_arguments_fail_explicitly(self):
        invalid = (
            {'width': 3},
            {'markers': 0},
            {'num_shared_prototypes': 0},
            {'num_task_prototypes': 0},
            {'temperature': 0},
            {'temperature': -0.25},
            {'temperature': float('inf')},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    PrototypeMarkerNet(**overrides)

    def test_factory_model_is_compatible_with_eight_way_tta(self):
        model = build_reconstruction_model({
            'architecture': 'prototype_marker',
            'width': 4,
            'markers': 4,
            'context': True,
            'num_shared_prototypes': 3,
            'num_task_prototypes': 2,
            'temperature': 0.25,
        }).eval()
        self.assertIsInstance(model, PrototypeMarkerNet)
        x = torch.rand(1, 1, 17, 23)
        with torch.no_grad():
            direct = model(x)
            unaugmented = predict(model, x, tta=1)
            averaged = predict(model, x, tta=8)
        torch.testing.assert_close(unaugmented, direct)
        self.assertEqual(averaged.shape, (1, 4, 17, 23))
        self.assertTrue(torch.isfinite(averaged).all().item())
        self.assertTrue(((averaged >= 0) & (averaged <= 1)).all().item())


if __name__ == '__main__':
    unittest.main()
