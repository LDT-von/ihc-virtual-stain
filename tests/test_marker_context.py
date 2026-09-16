import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

from src.data.roi_manifest import (MARKERS, PairedMarkers, build_manifest, digest,
                                    read_gray, validate_manifest)
from src.models.marker_context import MarkerContextNet, local_ssim, reconstruction_loss
from src.train_marker_context import (eval_command, infer_command, inverse_transform,
                                      load_model, predict, seed_all, train, transform)


class MarkerContextTests(unittest.TestCase):
    def setUp(self):
        seed_all(42)

    def test_local_ssim_matches_skimage(self):
        rng = np.random.default_rng(1)
        for scale in (1., .05, .001):
            x = rng.random((2, 4, 32, 35), dtype=np.float32)*scale
            y = rng.random(x.shape, dtype=np.float32)*scale
            actual = local_ssim(torch.from_numpy(x), torch.from_numpy(y)).numpy()
            expected = np.array([[structural_similarity(a, b, data_range=1) for a, b in zip(aa, bb)] for aa, bb in zip(x, y)])
            np.testing.assert_allclose(actual, expected, atol=3e-5)

    def test_ssim_identity_constant_and_shape(self):
        x = torch.full((1, 4, 32, 32), .3)
        torch.testing.assert_close(local_ssim(x, x), torch.ones(1, 4))
        with self.assertRaises(ValueError):
            local_ssim(x, x[..., :20])

    def test_shared_metric_uses_local_windows_and_rejects_empty(self):
        from src.metrics.ssim_psnr import MetricAggregator, ssim_gpu_single
        x, y = torch.rand(3, 32, 32), torch.rand(3, 32, 32)
        expected = np.mean([structural_similarity(a.numpy(), b.numpy(), data_range=1) for a, b in zip(x, y)])
        self.assertAlmostEqual(ssim_gpu_single(x*2-1, y*2-1), expected, places=5)
        metric = MetricAggregator()
        with self.assertRaises(ValueError):
            metric.result()
        metric.update(x*2-1, y*2-1)
        metric.update((x*2-1).unsqueeze(0), (y*2-1).unsqueeze(0))
        self.assertEqual(metric.n, 2)
        self.assertAlmostEqual(metric.result()['ssim'], expected, places=5)

    def test_tta_inverse_is_exact_and_unique(self):
        x = torch.arange(3*5).reshape(1, 1, 3, 5).float()/15
        variants = []
        for k in range(4):
            for flip in (False, True):
                y = transform(x, k, flip)
                torch.testing.assert_close(inverse_transform(y, k, flip), x, rtol=0, atol=0)
                variants.append(tuple(y.flatten().tolist()))
        self.assertEqual(len(set(variants)), 8)
        for n in (1, 4, 8):
            torch.testing.assert_close(predict(torch.nn.Identity(), x, n), x)

    def test_model_shape_bounds_and_all_heads_receive_gradient(self):
        net = MarkerContextNet(width=4)
        x = torch.rand(2, 1, 31, 35)
        y = net(x)
        self.assertEqual(y.shape, (2, 4, 31, 35))
        self.assertTrue(((y >= 0) & (y <= 1)).all())
        reconstruction_loss(y, torch.rand_like(y)).backward()
        for head in net.heads:
            self.assertGreater(head[-1].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))

    def test_legacy_tta_restores_coordinates_and_uses_eight_calls(self):
        from inference_8x_tta import tta_8x_forward
        from tta_eval import tta_8x
        class IdentityGenerator:
            def __init__(self):
                self.calls = 0
            def generator(self, x, condition):
                self.calls += 1
                return x
        x = torch.rand(1, 3, 13, 17)*2-1
        for fn in (tta_8x_forward, tta_8x):
            model = IdentityGenerator()
            torch.testing.assert_close(fn(model, x), x)
            self.assertEqual(model.calls, 8)

    def make_fixture(self, root):
        for marker in ('DAPI',)+MARKERS:
            folder = root/'train'/marker
            folder.mkdir(parents=True)
            for roi in range(7):
                arr = np.random.default_rng(roi).integers(0, 80, (256, 256), dtype=np.uint8)
                Image.fromarray(arr).save(folder/f'ROI{roi:03}_00_00.jpg', quality=95)
        return build_manifest(root, root/'manifest.json', val_rois=1, holdout_rois=1)

    def test_manifest_rejects_roi_overlap_even_with_new_checksum(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.make_fixture(root)
            manifest['splits']['val'] = [manifest['splits']['train'][0]]
            manifest['sha256'] = digest({k: v for k, v in manifest.items() if k != 'sha256'})
            with self.assertRaisesRegex(ValueError, 'leakage'):
                validate_manifest(manifest)
            with self.assertRaises(FileExistsError):
                build_manifest(root, root/'manifest.json')

    def test_augmentation_preserves_target_intensity_and_alignment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.make_fixture(root)
            ds = PairedMarkers(root, manifest['splits']['train'][:1], augment=True)
            expected = np.sort(ds.cache[0][0].ravel())
            for _ in range(8):
                x, y, _ = ds[0]
                torch.testing.assert_close(x.expand_as(y), y)
                np.testing.assert_array_equal(np.sort(x.mul(255).round().byte().numpy().ravel()), expected)

    def test_color_input_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'color.png'
            arr = np.zeros((8, 8, 3), dtype=np.uint8)
            arr[..., 0] = 255
            Image.fromarray(arr).save(path)
            with self.assertRaisesRegex(ValueError, 'Non-grayscale'):
                read_gray(path)

    def test_real_train_checkpoint_eval_infer_integration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_fixture(root)
            args = argparse.Namespace(data_root=str(root), manifest=str(root/'manifest.json'),
                output=str(root/'run'), seed=42, device='cpu', batch_size=2, width=4,
                epochs=1, lr=5e-4, no_context=False, no_cache=False,
                train_limit=2, val_limit=1, resume=None)
            train(args)
            model, ckpt = load_model(root/'run'/'best.pt', torch.device('cpu'))
            self.assertEqual(tuple(ckpt['run']['markers']), MARKERS)
            self.assertEqual(ckpt['steps'], 1)
            eval_args = argparse.Namespace(data_root=str(root), manifest=str(root/'manifest.json'),
                checkpoint=str(root/'run'/'best.pt'), seed=42, device='cpu', batch_size=2,
                split='holdout', limit=1, tta=1, jpeg_quality=95, output=str(root/'holdout.json'))
            eval_command(eval_args)
            self.assertEqual(json.loads((root/'holdout.json').read_text())['count'], 1)
            inference_input = root/'inputs'
            inference_input.mkdir()
            Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(inference_input/'ROI025_00_00.jpg')
            infer_args = argparse.Namespace(checkpoint=eval_args.checkpoint, input=str(inference_input),
                output=str(root/'submission'), device='cpu', seed=42, batch_size=2, tta=8, jpeg_quality=95)
            infer_command(infer_args)
            files = list((root/'submission'/'results'/'test').glob('*/*_fake.jpg'))
            self.assertEqual(len(files), 4)
            for file in files:
                with Image.open(file) as image:
                    self.assertEqual(image.size, (256, 256))
            with self.assertRaises(FileExistsError):
                infer_command(infer_args)
            # A completed run can resume safely without repeating an optimizer step.
            args.resume = str(root/'run'/'last.pt')
            train(args)
            refit = copy.deepcopy(args)
            refit.output = str(root/'refit')
            refit.resume = None
            refit.init_checkpoint = str(root/'run'/'best.pt')
            refit.full_data = True
            refit.train_limit = refit.val_limit = 0
            train(refit)
            self.assertTrue((root/'refit'/'final.pt').is_file())
            _, full_ckpt = load_model(root/'refit'/'final.pt', torch.device('cpu'))
            self.assertEqual(len(full_ckpt['run']['train_names']), 7)
            self.assertEqual(full_ckpt['run']['val_names'], [])
            self.assertIsNone(full_ckpt['metrics'])
            eval_args.checkpoint = str(root/'refit'/'final.pt')
            with self.assertRaisesRegex(ValueError, 'used for training'):
                eval_command(eval_args)
            refit.output = str(root/'invalid_clean_run')
            refit.full_data = False
            refit.init_checkpoint = str(root/'refit'/'final.pt')
            with self.assertRaisesRegex(ValueError, 'already saw'):
                train(refit)


if __name__ == '__main__':
    unittest.main()
