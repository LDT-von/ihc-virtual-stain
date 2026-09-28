"""Paired ROI comparison of two completed semifinal V7 selections.

This is a development-set comparison, never a prediction of platform score.
The same ROI inventory must appear in both selections. Results are grouped by
ROI so 140 neighbouring patches cannot masquerade as 140 independent cases.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def read_selection(path):
    selection = json.loads(Path(path).read_text(encoding="utf-8"))
    if selection.get("kind") != "semifinal_v7_cv_selection":
        raise ValueError(f"Not a V7 selection: {path}")
    selected = selection["selected_metrics"]
    if selected["epoch"] != selection["selected_epoch"]:
        raise ValueError(f"Selected epoch mismatch: {path}")
    if set(selected["rois"]) != set(selection["all_rois"]):
        raise ValueError(f"ROI coverage mismatch: {path}")
    if not all(math.isfinite(selected[key]) for key in ("ssim", "psnr")):
        raise ValueError(f"Non-finite score: {path}")
    return selection


def bootstrap_roi(deltas, seed, repeats=10000):
    values = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(repeats, len(values)))
    means = values[draws].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Baseline selection JSON")
    parser.add_argument("--candidate", required=True, help="Candidate selection JSON")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    baseline = read_selection(args.baseline)
    candidate = read_selection(args.candidate)
    for field in ("split_sha256", "all_rois", "source_sha256",
                  "environment_source_sha256"):
        if baseline[field] != candidate[field]:
            raise ValueError(f"Unmatched experiment provenance: {field}")
    for field in ("folds", "seed", "tta"):
        if baseline["recipe"][field] != candidate["recipe"][field]:
            raise ValueError(f"Different fold assignment or inference protocol: {field}")
    rois = sorted(baseline["all_rois"])
    base = baseline["selected_metrics"]
    cand = candidate["selected_metrics"]
    if base["count"] != cand["count"] or set(base["markers"]) != set(cand["markers"]):
        raise ValueError("Different validation image count or marker set")
    report = {"baseline": str(Path(args.baseline).resolve()),
              "candidate": str(Path(args.candidate).resolve()),
              "roi_count": len(rois), "baseline_epoch": baseline["selected_epoch"],
              "candidate_epoch": candidate["selected_epoch"],
              "warning": "Development OOF evidence; recipe and epoch were selected on these ROIs."}
    for metric in ("ssim", "psnr"):
        deltas = [cand["rois"][roi][metric] - base["rois"][roi][metric]
                  for roi in rois]
        report[metric] = {
            "pooled_delta": cand[metric] - base[metric],
            "roi_mean_delta": float(np.mean(deltas)),
            "roi_median_delta": float(np.median(deltas)),
            "roi_bootstrap_95pct": bootstrap_roi(deltas, args.seed),
            "rois_improved": sum(delta > 0 for delta in deltas),
            "rois_worsened": sum(delta < 0 for delta in deltas),
            "per_roi_delta": dict(zip(rois, map(float, deltas))),
            "per_marker_pooled_delta": {
                marker: cand["markers"][marker][metric] - base["markers"][marker][metric]
                for marker in sorted(base["markers"])
            },
        }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
