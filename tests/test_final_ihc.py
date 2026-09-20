import argparse
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.package_final import package
from src.data.dataset import build_transforms
from src.data.roi_manifest import MARKERS, build_manifest
from src.models.flow_matching import FlowMatching, FlowMatchingConfig
from src.models.marker_specific import MarkerSpecificNet
from src.train_marker_context import train, load_model, infer_command, seed_all


class ConstantOracle(torch.nn.Module):
    def forward(self, x, t, condition):
        return condition


class FinalIHCTests(unittest.TestCase):
    def setUp(self):
        seed_all(42)

    def test_reverse_flow_reaches_analytic_endpoint_for_both_solvers(self):
        noise, target = torch.randn(2, 3, 8, 9), torch.rand(2, 3, 8, 9)
        for method, sigma in [('optimal_transport', 0.), ('independent', .1)]:
            for solver in ('euler', 'heun'):
                for steps in (1, 4, 15):
                    fm = FlowMatching(ConstantOracle(), FlowMatchingConfig(method=method, sigma_min=sigma))
                    result = fm.sample(target-noise, steps, noise, solver)
                    torch.testing.assert_close(result, (1-sigma)*target+sigma*noise)

    def test_training_velocity_direction_matches_sampler(self):
        x, condition = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)
        fm = FlowMatching(ConstantOracle(), FlowMatchingConfig())
        torch.manual_seed(3)
        t, noise = torch.rand(2), torch.randn_like(x)
        torch.manual_seed(3)
        loss, xt, _ = fm.forward_train(x, condition)
        torch.testing.assert_close(xt, (1-t[:, None, None, None])*x+t[:, None, None, None]*noise)
        torch.testing.assert_close(loss, (condition-(x-noise)).square().mean())

    def test_flow_rejects_invalid_solver_and_steps(self):
        fm = FlowMatching(ConstantOracle(), FlowMatchingConfig())
        x = torch.zeros(1, 3, 8, 8)
        for steps in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                fm.sample(x, num_steps=steps)
        with self.assertRaises(ValueError):
            fm.sample(x, solver='typo')

    def test_legacy_pair_augmentation_preserves_truth(self):
        arr = np.random.default_rng(4).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        transform = build_transforms(32, True)
        for _ in range(20):
            result = transform(image=arr, ihc=arr)
            torch.testing.assert_close(result['image'], result['ihc'])
            restored = ((result['ihc']+1)*127.5).round().byte().permute(1, 2, 0).numpy()
            np.testing.assert_array_equal(np.sort(restored.reshape(-1, 3), axis=0),
                                          np.sort(arr.reshape(-1, 3), axis=0))

    def test_private_decoder_gradients_and_odd_size(self):
        net = MarkerSpecificNet(width=4)
        result = net(torch.rand(2, 1, 31, 35))
        self.assertEqual(tuple(result.shape), (2, 4, 31, 35))
        self.assertTrue(((result >= 0) & (result <= 1)).all())
        result[:, 0].mean().backward()
        self.assertGreater(net.stem.weight.grad.abs().sum().item(), 0)
        self.assertGreater(net.decoders[0].head.weight.grad.abs().sum().item(), 0)
        for decoder in net.decoders[1:]:
            self.assertTrue(all(p.grad is None or p.grad.count_nonzero() == 0 for p in decoder.parameters()))
        net.zero_grad(set_to_none=True)
        net(torch.rand(1, 1, 32, 32)).mean().backward()
        for decoder in net.decoders:
            self.assertGreater(decoder.head.weight.grad.abs().sum().item(), 0)

    def test_final_train_reload_inference_and_strict_package(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for marker in ('DAPI',)+MARKERS:
                folder = root/'train'/marker
                folder.mkdir(parents=True)
                for roi in range(4):
                    a = np.random.default_rng(roi).integers(0, 80, (256, 256), dtype=np.uint8)
                    Image.fromarray(a).save(folder/f'ROI{roi:03}_00_00.jpg')
            build_manifest(root, root/'split.json', val_rois=1, holdout_rois=1)
            args = argparse.Namespace(data_root=str(root), manifest=str(root/'split.json'),
                output=str(root/'run'), seed=42, device='cpu', batch_size=1, width=4,
                epochs=1, lr=5e-4, no_context=False, no_cache=False, architecture='marker_specific',
                val_jpeg_quality=95, train_limit=1, val_limit=1, resume=None)
            train(args)
            net, ck = load_model(root/'run'/'best.pt', torch.device('cpu'))
            self.assertIsInstance(net, MarkerSpecificNet)
            self.assertEqual(ck['metrics']['jpeg_quality'], 95)
            args.resume = str(root/'run'/'last.pt')
            train(args)
            inputs = root/'test'/'DAPI'
            inputs.mkdir(parents=True)
            Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(inputs/'ROI025_00_00.jpg')
            infer_command(argparse.Namespace(checkpoint=args.resume, input=str(inputs),
                output=str(root/'pred'), device='cpu', seed=42, batch_size=1, tta=4, jpeg_quality=95))
            report = package(inputs, root/'pred', root/'submission.zip')
            self.assertEqual(report['total_images'], 4)
            (root/'pred'/'results'/'test'/'CD68'/'ROI025_00_00_fake.jpg').unlink()
            with self.assertRaisesRegex(ValueError, 'missing=1'):
                package(inputs, root/'pred', root/'bad.zip')
            self.assertFalse((root/'bad.zip').exists())

    def test_full_trainer_latest_contains_optimizer_and_resumes_real_steps(self):
        import scripts.train_full as full

        class TinyVelocity(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1)
            def forward(self, x, t, condition):
                return self.conv(x)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for marker in ('DAPI', 'CD68'):
                folder = root/'train'/marker
                folder.mkdir(parents=True)
                Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(folder/'ROI000_00_00.jpg')
            args = argparse.Namespace(marker='CD68', data_root=str(root), epochs=1,
                                      batch_size=2, lr=5e-4, resume_from=None)
            # Use a fresh output root while exercising the real optimizer/checkpoint loop.
            import os
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(full, 'parse_args', return_value=args), \
                     patch.object(full, 'build_model', side_effect=lambda **kw: TinyVelocity()), \
                     patch.object(torch.cuda, 'is_available', return_value=False):
                    full.main()
                    latest = next((root/'checkpoints').glob('*/latest.pt'))
                    ck = torch.load(latest, weights_only=False)
                    self.assertIn('optimizer', ck)
                    self.assertEqual(ck['global_step'], 1)
                    args.resume_from, args.epochs = str(latest), 2
                    full.main()
                    resumed = torch.load(latest, weights_only=False)
                    self.assertEqual(resumed['global_step'], 2)
                    self.assertEqual(resumed['epoch'], 1)
                    self.assertEqual(next(iter(resumed['optimizer']['state'].values()))['step'].item(), 2)
            finally:
                os.chdir(previous)

    def test_fm_inference_uses_cli_solver_and_official_names(self):
        import scripts.infer_fm_test as inference
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = root/'test'/'DAPI'
            inputs.mkdir(parents=True)
            Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).save(inputs/'ROI025_00_00.jpg')
            model = torch.nn.Conv2d(3, 3, 1)
            config = {'model': {'in_channels': 3, 'cond_channels': 3, 'base_channels': 4,
                'channel_mults': [1], 'num_res_blocks': 1, 'attention_resolutions': []},
                'defaults': {'marker': 'CD68'}}
            torch.save({'model': model.state_dict(), 'cfg': config}, root/'model.pt')
            args = argparse.Namespace(ckpt=str(root/'model.pt'), marker='CD68', data_root=str(root),
                output_dir=str(root/'pred'), batch_size=1, jpeg_quality=95, num_steps=3, solver='heun', seed=42)
            with patch.object(inference, 'parse_args', return_value=args), \
                 patch.object(inference, 'build_model', return_value=model), \
                 patch.object(inference.FlowMatching, 'sample', return_value=torch.zeros(1,3,256,256)) as sample, \
                 patch.object(torch.cuda, 'is_available', return_value=False):
                inference.main()
                self.assertEqual(sample.call_args.kwargs, {'num_steps': 3, 'solver': 'heun'})
            self.assertTrue((root/'pred/results/test/CD68/ROI025_00_00_fake.jpg').is_file())


if __name__ == '__main__':
    unittest.main()
