"""Synthetic end-to-end regression for the semifinal V7 single-marker protocol."""

import argparse
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

import train_semifinal_v7 as v7
from src.data.roi_manifest import MARKERS, SEMIFINAL_SEED, build_manifest


class SemifinalV7Tests(unittest.TestCase):
    @staticmethod
    def gray_image(roi):
        yy, xx = np.indices((256, 256), dtype=np.uint16)
        return ((17 * (xx // 8) + 11 * (yy // 8) + 19 * roi) % 200 + 25).astype(np.uint8)

    @staticmethod
    def save_jpeg(path, gray):
        path.parent.mkdir(parents=True, exist_ok=True)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        Image.fromarray(rgb, mode="RGB").save(
            path, format="JPEG", quality=100, subsampling=0
        )

    def make_fixture(self, root):
        yy, xx = np.indices((256, 256), dtype=np.uint16)
        for roi in range(5):
            name = f"ROI{roi:03d}_00_00.jpg"
            dapi = self.gray_image(roi)
            targets = {
                "HLA-DR": ((dapi.astype(np.uint16) // 2
                            + 2 * ((xx + roi) % 31)) % 256).astype(np.uint8),
                "CD68": np.where((xx // 16 + yy // 16 + roi) % 3 == 0,
                                  180, dapi // 3).astype(np.uint8),
                "CD45RO": ((3 * (yy // 11) + xx // 7 + 23 * roi) % 180
                            + 20).astype(np.uint8),
                "Vimentin": ((xx // 4 + 2 * (yy // 9) + 29 * roi) % 150
                               + 40).astype(np.uint8),
            }
            self.assertEqual(tuple(targets), MARKERS)
            self.assertEqual(len({image.tobytes() for image in targets.values()}), 4)
            for marker, image in [("DAPI", dapi), *targets.items()]:
                self.save_jpeg(root / "train" / marker / name, image)
        return build_manifest(
            root, root / "manifest.json", val_rois=1, holdout_rois=1
        )

    @staticmethod
    def fit_args(root, output, *, stage="dev", fold=0, selection=None, device="cpu",
                 batch_size=2):
        return argparse.Namespace(
            stage=stage, data_root=str(root), manifest=str(root / "manifest.json"),
            output=str(output), selection=str(selection) if selection else None,
            fold=fold, folds=2, augmentation="geometry", loss="normalized",
            architecture="marker_nafnet", target_marker="CD68", width=4,
            num_shared_prototypes=8, num_task_prototypes=4,
            prototype_temperature=0.25, prototype_diversity_weight=0.001,
            epochs=1, eval_every=1, batch_size=batch_size, lr=3e-4,
            cd68_weight=3.0, warmup_epochs=0, weaken_start_epoch=200, tta=1,
            seed=SEMIFINAL_SEED, device=device, no_cache=False, resume=None,
        )

    @staticmethod
    def read_json(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def test_two_folds_selection_refit_infer_and_rejections(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest = self.make_fixture(root)
                expected_names = set(sum(manifest["splits"].values(), []))
                expected_rois = {name.split("_")[0] for name in expected_names}
                self.assertEqual(len(expected_names), 5)
                self.assertEqual(len(expected_rois), 5)

                directories = [root / f"fold_{fold}" for fold in range(2)]
                validation_rois = set()
                for fold, directory in enumerate(directories):
                    v7.fit(self.fit_args(root, directory, fold=fold))
                    run = self.read_json(directory / "run.json")
                    history = [json.loads(line) for line in
                               (directory / "history.jsonl").read_text(
                                   encoding="utf-8").splitlines()]
                    self.assertEqual(run["markers"], ["CD68"])
                    self.assertEqual(run["model_config"]["markers"], 1)
                    self.assertEqual(run["recipe"]["target_marker"], "CD68")
                    self.assertEqual(run["source_sha256"], v7.source_hashes())
                    self.assertEqual(run["split_sha256"], manifest["sha256"])
                    self.assertEqual(set(run["train_names"]) | set(run["val_names"]),
                                     expected_names)
                    self.assertFalse(set(run["train_names"]) & set(run["val_names"]))
                    self.assertFalse(set(run["train_rois"]) & set(run["val_rois"]))
                    self.assertEqual(set(run["train_rois"]) | set(run["val_rois"]),
                                     expected_rois)
                    self.assertFalse(validation_rois & set(run["val_rois"]))
                    validation_rois.update(run["val_rois"])
                    self.assertEqual(len(history), 1)
                    self.assertEqual(history[0]["epoch"], 1)
                    self.assertEqual(history[0]["validation"]["count"],
                                     len(run["val_names"]))
                    self.assertEqual(set(history[0]["validation"]["markers"]), {"CD68"})
                    self.assertTrue((directory / "eval_epoch_001.pt").is_file())
                self.assertEqual(validation_rois, expected_rois)

                # A completed resume may rewrite identical history, but not
                # checkpoint bytes or the successful update count.
                last_path = directories[0] / "last.pt"
                before_hash = hashlib.sha256(last_path.read_bytes()).hexdigest()
                before = torch.load(last_path, map_location="cpu", weights_only=False)
                resumed = self.fit_args(root, directories[0], fold=0)
                resumed.resume = str(last_path)
                v7.fit(resumed)
                after = torch.load(last_path, map_location="cpu", weights_only=False)
                self.assertEqual(hashlib.sha256(last_path.read_bytes()).hexdigest(),
                                 before_hash)
                self.assertEqual((after["steps"], after["epoch"]),
                                 (before["steps"], before["epoch"]))
                self.assertEqual(len((directories[0] / "history.jsonl").read_text(
                    encoding="utf-8").splitlines()), 1)

                with self.assertRaisesRegex(
                    ValueError, "Exactly one completed run per fold"
                ):
                    v7.summarize(argparse.Namespace(
                        runs=[str(directories[0])],
                        output=str(root / "incomplete.json"),
                    ))
                self.assertFalse((root / "incomplete.json").exists())

                selection_path = root / "selection.json"
                v7.summarize(argparse.Namespace(
                    runs=[str(path) for path in directories],
                    output=str(selection_path),
                ))
                selection = self.read_json(selection_path)
                self.assertEqual(selection["markers"], ["CD68"])
                self.assertEqual(selection["all_rois"], sorted(expected_rois))
                self.assertEqual(selection["source_sha256"], v7.source_hashes())
                self.assertEqual(selection["selected_epoch"], 1)
                self.assertEqual(selection["selected_metrics"]["epoch"], 1)
                self.assertEqual(selection["epoch_table"][0]["count"], 5)
                self.assertEqual(set(selection["epoch_table"][0]["markers"]), {"CD68"})

                final_args = self.fit_args(
                    root, root / "final_run", stage="final", fold=None,
                    selection=selection_path,
                )
                missing = copy.deepcopy(final_args)
                missing.selection = None
                missing.output = str(root / "missing_selection")
                with self.assertRaisesRegex(ValueError, "requires --selection"):
                    v7.fit(missing)
                self.assertFalse((root / "missing_selection").exists())

                wrong_marker = copy.deepcopy(final_args)
                wrong_marker.target_marker = "HLA-DR"
                wrong_marker.output = str(root / "wrong_marker")
                with self.assertRaisesRegex(ValueError, "Selection does not match"):
                    v7.fit(wrong_marker)
                self.assertFalse((root / "wrong_marker").exists())

                v7.fit(final_args)
                final_run = self.read_json(root / "final_run" / "run.json")
                final_checkpoint = torch.load(
                    root / "final_run" / "final.pt",
                    map_location="cpu", weights_only=False,
                )
                self.assertEqual(final_run["markers"], ["CD68"])
                self.assertEqual(set(final_run["train_names"]), expected_names)
                self.assertEqual(final_run["val_names"], [])
                self.assertEqual(final_run["stopped_at_epoch"],
                                 selection["selected_epoch"])
                self.assertEqual(final_checkpoint["epoch"], selection["selected_epoch"])
                self.assertEqual(final_checkpoint["run"]["model_config"]["markers"], 1)

                test_input = root / "test" / "DAPI"
                self.save_jpeg(test_input / "ROI900_00_00.jpg",
                               self.gray_image(900))
                submission = root / "submission"
                v7.infer(argparse.Namespace(
                    checkpoint=str(root / "final_run" / "final.pt"),
                    input=str(test_input), output=str(submission),
                    device="cpu", batch_size=2,
                ))
                marker_root = submission / "results" / "test"
                self.assertEqual([path.name for path in marker_root.iterdir()], ["CD68"])
                predictions = list((marker_root / "CD68").glob("*.jpg"))
                self.assertEqual([path.name for path in predictions],
                                 ["ROI900_00_00_fake.jpg"])
                with Image.open(predictions[0]) as image:
                    self.assertEqual((image.format, image.mode, image.size),
                                     ("JPEG", "RGB", (256, 256)))
                    channels = image.split()
                    self.assertEqual(channels[0].tobytes(), channels[1].tobytes())
                    self.assertEqual(channels[1].tobytes(), channels[2].tobytes())
                provenance = self.read_json(submission / "provenance.json")
                self.assertEqual(provenance["input_count"], 1)
                self.assertEqual(provenance["markers"], ["CD68"])
                self.assertEqual(provenance["tta"], 1)
                self.assertEqual(provenance["run"]["recipe"]["target_marker"], "CD68")
        finally:
            torch.set_num_threads(old_threads)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for FP16 overflow test")
    def test_fp16_overflow_skips_one_batch_then_updates_and_preserves_ema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_fixture(root)
            output = root / "cuda_fold"
            args = self.fit_args(
                root, output, fold=0, device="cuda", batch_size=1
            )
            original_factory = v7.build_reconstruction_model
            hook_calls = []

            def factory_with_one_overflow(config):
                model = original_factory(config)

                def overflow_once(gradient):
                    hook_calls.append(1)
                    if len(hook_calls) == 1:
                        return torch.full_like(gradient, float("inf"))
                    return gradient

                model.intro.weight.register_hook(overflow_once)
                return model

            with mock.patch.object(torch.cuda, "is_bf16_supported", return_value=False):
                with mock.patch.object(v7, "build_reconstruction_model",
                                       side_effect=factory_with_one_overflow):
                    v7.fit(args)

            run = self.read_json(output / "run.json")
            history = [json.loads(line) for line in
                       (output / "history.jsonl").read_text(
                           encoding="utf-8").splitlines()]
            checkpoint = torch.load(
                output / "last.pt", map_location="cpu", weights_only=False
            )
            self.assertGreater(len(hook_calls), 1)
            self.assertEqual(history[0]["skipped_overflow_batches"], 1)
            self.assertEqual(checkpoint["steps"], len(run["train_names"]) - 1)
            self.assertGreater(checkpoint["steps"], 0)
            for state in (checkpoint["model"], checkpoint["ema"]):
                self.assertTrue(all(torch.isfinite(tensor).all().item()
                                    for tensor in state.values()))
