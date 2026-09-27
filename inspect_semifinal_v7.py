"""Create worst-patch contact sheets from a completed semifinal V7 CV selection.

This script does no work on import. Run it only after all development folds
and the selection JSON exist. The images correspond to the globally selected
epoch, not each fold's independently best checkpoint.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.data.roi_manifest import MARKERS, read_gray
from src.models.marker_context import local_ssim
from src.train_marker_context import amp_context, jpeg_roundtrip, load_model, predict


def inspect(args):
    selection = json.loads(Path(args.selection).read_text(encoding='utf-8'))
    if selection.get('kind') != 'semifinal_v7_cv_selection':
        raise ValueError('Expected a semifinal V7 cross-validation selection')
    epoch = selection['selected_epoch']
    tta = selection['recipe']['tta']
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Inspection output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    chosen = {marker: [] for marker in MARKERS}
    expected_scores = {}
    runs = []
    for directory_name in selection['fold_directories']:
        directory = Path(directory_name)
        run = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
        if (run['kind'] != 'semifinal_v7_dev'
                or run['split_sha256'] != selection['split_sha256']
                or run['source_sha256'] != selection['source_sha256']
                or run['recipe'] != selection['recipe']):
            raise ValueError(f'Fold does not match selection: {directory}')
        metrics = json.loads((directory / f'eval_epoch_{epoch:03d}.json').read_text(
            encoding='utf-8'))
        if (metrics['count'] != len(run['val_names'])
                or {row['name'] for row in metrics['rows']} != set(run['val_names'])):
            raise ValueError(f'Incomplete selected-epoch rows: {directory}')
        for row in metrics['rows']:
            expected_scores[(directory, row['name'])] = row['ssim']
            for index, marker in enumerate(MARKERS):
                chosen[marker].append((float(row['ssim'][index]), directory, row['name']))
        runs.append((directory, run))
    if len(runs) != selection['recipe']['folds']:
        raise ValueError('Missing folds')

    picked = {marker: sorted(rows, key=lambda item: (item[0], item[2]))[:args.top_k]
              for marker, rows in chosen.items()}
    required = {}
    for rows in picked.values():
        for _, directory, name in rows:
            required.setdefault(directory, set()).add(name)
    device = torch.device(args.device)
    predictions = {}
    for directory, names in required.items():
        checkpoint_path = directory / f'eval_epoch_{epoch:03d}.pt'
        model, checkpoint = load_model(checkpoint_path, device)
        if (checkpoint.get('epoch') != epoch
                or checkpoint['run']['source_sha256'] != selection['source_sha256']):
            raise ValueError(f'Wrong selected-epoch checkpoint: {checkpoint_path}')
        for name in sorted(names):
            dapi = read_gray(Path(args.data_root) / 'train' / 'DAPI' / name)
            truth = np.stack([read_gray(Path(args.data_root) / 'train' / marker / name)
                              for marker in MARKERS])
            x = torch.from_numpy(dapi.copy()[None, None]).float().to(device) / 255
            with torch.no_grad(), amp_context(device):
                prediction = predict(model, x, tta)
            prediction = jpeg_roundtrip(prediction)[0].mul(255).round().to(
                torch.uint8).cpu().numpy()
            actual_ssim = local_ssim(
                torch.from_numpy(prediction[None].astype(np.float32) / 255),
                torch.from_numpy(truth[None].astype(np.float32) / 255),
            )[0].tolist()
            if any(abs(actual - recorded) > 0.002 for actual, recorded in zip(
                    actual_ssim, expected_scores[(directory, name)])):
                raise ValueError(f'Images or inference differ from recorded CV metrics: {name}')
            predictions[(directory, name)] = prediction
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    report = {'kind': 'semifinal_v7_visual_inspection', 'selected_epoch': epoch,
              'tta': tta, 'selection_sha256': hashlib.sha256(
                  Path(args.selection).read_bytes()).hexdigest(), 'worst': {}}
    tile, label_height = 256, 34
    for marker_index, marker in enumerate(MARKERS):
        rows = picked[marker]
        canvas = Image.new('RGB', (tile * 4, (tile + label_height) * len(rows)), 'white')
        draw = ImageDraw.Draw(canvas)
        report['worst'][marker] = []
        for row_index, (ssim, directory, name) in enumerate(rows):
            dapi = read_gray(Path(args.data_root) / 'train' / 'DAPI' / name)
            truth = read_gray(Path(args.data_root) / 'train' / marker / name)
            prediction = predictions[(directory, name)][marker_index]
            error = np.abs(truth.astype(np.int16) - prediction.astype(np.int16))
            truth_p99 = float(np.percentile(truth, 99))
            prediction_p99 = float(np.percentile(prediction, 99))
            # Error is displayed on the same 0..255 scale as the other panels.
            base_y = row_index * (tile + label_height)
            for column, image in enumerate((dapi, truth, prediction,
                                            error.astype(np.uint8))):
                canvas.paste(Image.fromarray(image).convert('RGB'),
                             (column * tile, base_y + label_height))
            draw.text((4, base_y + 8), f'{name}  SSIM={ssim:.4f}  fold={directory.name}',
                      fill='black')
            draw.text((4, base_y + 20),
                      f'GT p99={truth_p99:.0f}  prediction p99={prediction_p99:.0f}',
                      fill='black')
            report['worst'][marker].append({'name': name, 'ssim': ssim,
                                            'gt_p99': truth_p99,
                                            'prediction_p99': prediction_p99,
                                            'fold_directory': str(directory)})
        canvas.save(output / f'{marker}_worst.png')
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                        encoding='utf-8')
    print(f'Wrote {len(MARKERS)} contact sheets to {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--top-k', type=int, default=6)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.top_k < 1:
        raise ValueError('top-k must be positive')
    inspect(args)


if __name__ == '__main__':
    main()
