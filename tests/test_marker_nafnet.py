"""Focused regression checks for the DAPI-only MarkerNAFNet pipeline."""

import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from src.data.marker_subset import MarkerSubsetDataset
from src.data.roi_manifest import MARKERS, SEMIFINAL_SEED
from src.models.marker_context import local_ssim
from src.models.marker_nafnet import MarkerNAFNet
from src.train_marker_context import (encode_rgb_jpeg, evaluate, jpeg_roundtrip,
                                      load_model, predict)
from train_semifinal_v7 import infer, source_hashes


class _FourMarkerSample:
    def __init__(self):
        self.aug_strength = 0.25

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        image = torch.full((1, 16, 19), 0.5)
        target = torch.stack(
            [torch.full((16, 19), float(channel) / 4) for channel in range(4)]
        )
        return image, target, "ROI001_00_00.jpg"


class _FixedPrediction(torch.nn.Module):
    def __init__(self, prediction):
        super().__init__()
        self.register_buffer("prediction", prediction)

    def forward(self, image):
        return self.prediction.expand(image.shape[0], -1, -1, -1)


class MarkerNAFNetTests(unittest.TestCase):
    def test_single_and_four_marker_rectangles_support_tta_and_gradients(self):
        torch.manual_seed(13)
        for markers, height, width in ((1, 16, 19), (4, 17, 23)):
            with self.subTest(markers=markers, size=(height, width)):
                model = MarkerNAFNet(markers=markers, width=4)
                image = torch.rand(1, 1, height, width, requires_grad=True)
                prediction = model(image)
                self.assertEqual(prediction.shape, (1, markers, height, width))
                self.assertTrue(torch.isfinite(prediction).all().item())
                self.assertTrue(((prediction >= 0) & (prediction <= 1)).all().item())

                (prediction - torch.rand_like(prediction)).square().mean().backward()
                self.assertIsNotNone(image.grad)
                self.assertGreater(image.grad.abs().sum().item(), 0)
                for head in model.marker_heads:
                    gradient = head[-1].weight.grad
                    self.assertIsNotNone(gradient)
                    self.assertTrue(torch.isfinite(gradient).all().item())
                    self.assertGreater(gradient.abs().sum().item(), 0)

                model.eval()
                with torch.no_grad():
                    unaugmented = predict(model, image.detach(), tta=1)
                    averaged = predict(model, image.detach(), tta=8)
                    direct = model(image.detach())
                torch.testing.assert_close(unaugmented, direct)
                self.assertEqual(averaged.shape, prediction.shape)
                self.assertTrue(torch.isfinite(averaged).all().item())
                self.assertTrue(((averaged >= 0) & (averaged <= 1)).all().item())

    def test_constructor_and_input_contracts(self):
        for kwargs in ({"markers": 0}, {"width": 0}, {"markers": True}, {"width": 2.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MarkerNAFNet(**kwargs)
        model = MarkerNAFNet(markers=1, width=4)
        with self.assertRaises(ValueError):
            model(torch.rand(1, 2, 16, 19))
        with self.assertRaises(ValueError):
            model(torch.rand(1, 1, 15, 19))
        with self.assertRaises(TypeError):
            model(torch.zeros(1, 1, 16, 19, dtype=torch.uint8))

    def test_marker_subset_selects_targets_and_proxies_augmentation(self):
        base = _FourMarkerSample()
        selected = MarkerSubsetDataset(base, (MARKERS[1], MARKERS[3]))
        image, target, name = selected[0]
        self.assertEqual(len(selected), 1)
        self.assertEqual(tuple(selected.markers), (MARKERS[1], MARKERS[3]))
        self.assertEqual(image.shape, (1, 16, 19))
        self.assertEqual(name, "ROI001_00_00.jpg")
        torch.testing.assert_close(target, base[0][1][[1, 3]])
        self.assertEqual(selected.aug_strength, 0.25)
        selected.aug_strength = 0.7
        self.assertEqual(base.aug_strength, 0.7)
        base.aug_strength = 0.4
        self.assertEqual(selected.aug_strength, 0.4)

        for names in ((MARKERS[2], MARKERS[0]), (MARKERS[0], MARKERS[0]), (),
                      ("unknown-marker",)):
            with self.subTest(names=names), self.assertRaises(ValueError):
                MarkerSubsetDataset(base, names)

    def test_jpeg_roundtrip_rejects_nonfinite_predictions(self):
        prediction = torch.full((1, 1, 16, 19), 0.5)
        prediction[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "(?i)finite|nan"):
            jpeg_roundtrip(prediction)
    def test_evaluate_uses_decoded_rgb_submission_pixels(self):
        height, width = 16, 19
        values = (torch.arange(height * width).reshape(height, width) * 73) % 256
        prediction = ((values.float() + 0.37) / 256).reshape(1, 1, height, width)
        target = torch.full_like(prediction, 0.3)
        quantized = prediction[0, 0].mul(255).round().to(torch.uint8).numpy()
        with Image.open(io.BytesIO(encode_rgb_jpeg(quantized))) as image:
            rgb = np.asarray(image.convert("RGB"))
        self.assertTrue(np.array_equal(rgb[..., 0], rgb[..., 1]))
        self.assertTrue(np.array_equal(rgb[..., 0], rgb[..., 2]))
        expected = torch.from_numpy(rgb[..., 0].copy()).float().reshape(1, 1, height, width) / 255
        torch.testing.assert_close(jpeg_roundtrip(prediction), expected, rtol=0, atol=0)

        sample = [(torch.zeros_like(prediction), target, ["ROI001_00_00.jpg"])]
        metrics = evaluate(_FixedPrediction(prediction), sample, torch.device("cpu"),
                           tta=1, marker_names=(MARKERS[0],))
        expected_ssim = local_ssim(expected, target).item()
        expected_mse = (expected - target).square().mean().item()
        expected_psnr = -10 * math.log10(max(expected_mse, 1e-12))
        self.assertEqual(set(metrics["markers"]), {MARKERS[0]})
        self.assertEqual(metrics["count"], 1)
        self.assertAlmostEqual(metrics["rows"][0]["ssim"][0], expected_ssim, places=6)
        self.assertAlmostEqual(metrics["rows"][0]["psnr"][0], expected_psnr, places=5)

    def test_single_marker_checkpoint_loads_and_infer_writes_only_that_marker(self):
        marker = MARKERS[1]
        model = MarkerNAFNet(markers=1, width=4)
        run = {
            "kind": "semifinal_v7_final",
            "source_sha256": source_hashes(),
            "stopped_at_epoch": 1,
            "model_config": {"architecture": "marker_nafnet", "markers": 1, "width": 4},
            "markers": [marker],
            "recipe": {"seed": SEMIFINAL_SEED, "tta": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "single.pt"
            torch.save({"format": 1, "run": run, "epoch": 1, "ema": model.state_dict()},
                       checkpoint_path)
            restored, checkpoint = load_model(checkpoint_path, torch.device("cpu"))
            self.assertIsInstance(restored, MarkerNAFNet)
            self.assertEqual(restored.markers, 1)
            self.assertEqual(checkpoint["run"]["markers"], [marker])

            inputs = root / "inputs"
            inputs.mkdir()
            gray = np.arange(256 * 256, dtype=np.uint32).reshape(256, 256).astype(np.uint8)
            Image.fromarray(gray).save(inputs / "ROI001_00_00.jpg", quality=100)
            output = root / "submission"
            args = SimpleNamespace(checkpoint=checkpoint_path, input=inputs, output=output,
                                   device="cpu", batch_size=1, seed=0, tta=8)
            infer(args)
            self.assertEqual(args.seed, SEMIFINAL_SEED)
            self.assertEqual(args.tta, 1)
            test_root = output / "results" / "test"
            self.assertEqual({path.name for path in test_root.iterdir()}, {marker})
            files = list((test_root / marker).glob("*.jpg"))
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].name, "ROI001_00_00_fake.jpg")
            with Image.open(files[0]) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (256, 256))
            provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(provenance["markers"], [marker])

            nan_state = {key: value.clone() for key, value in model.state_dict().items()}
            nan_state["marker_heads.0.1.bias"].fill_(float("nan"))
            nan_path = root / "nan_output.pt"
            torch.save({"format": 1, "run": run, "epoch": 1, "ema": nan_state}, nan_path)
            nan_output = root / "nan_submission"
            nan_args = SimpleNamespace(checkpoint=nan_path, input=inputs, output=nan_output,
                                       device="cpu", batch_size=1, seed=0, tta=8)
            with self.assertRaisesRegex(ValueError, "(?i)finite|nan"):
                infer(nan_args)
            self.assertFalse(list((nan_output / "results" / "test" / marker).glob("*.jpg")))
            bad_run = {**run, "model_config": {**run["model_config"], "markers": 2}}
            bad_path = root / "bad_count.pt"
            torch.save({"format": 1, "run": bad_run, "epoch": 1,
                        "ema": model.state_dict()}, bad_path)
            with self.assertRaisesRegex(ValueError, "marker order"):
                load_model(bad_path, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
