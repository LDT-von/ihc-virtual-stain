"""用 best.pt 跑测试集 DAPI，输出 4 个 marker 的 JPG，再打包 submission.zip

用途：把 latest best.pt 的预测结果打包为官方格式 ZIP。
格式：results/test/<marker>/<name>_fake.jpg
"""
import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys
ROOT = Path(r'E:\aic\final-ihc')
sys.path.insert(0, str(ROOT))
from src.data.roi_manifest import MARKERS, read_gray
from src.train_marker_context import (build_reconstruction_model, predict,
                                       jpeg_roundtrip, amp_context, OFFICIAL_JPEG_QUALITY)


def infer_zip(ckpt_path: Path, test_dir: Path, output_zip: Path,
              tta: int = 4, batch_size: int = 4, device: str = 'cuda'):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if tuple(ckpt['run']['markers']) != tuple(MARKERS):
        raise ValueError(f"checkpoint markers {ckpt['run']['markers']} != {MARKERS}")

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = build_reconstruction_model(ckpt['run']['model_config']).to(dev)
    model.load_state_dict(ckpt['ema'], strict=True)
    model.eval()
    print(f'[load] {ckpt_path.name}  format={ckpt.get("format")}  '
          f'width={ckpt["run"]["model_config"].get("width")}  tta={tta}')

    inputs = sorted(test_dir.glob('*.jpg'))
    if not inputs:
        raise ValueError(f'No jpg in {test_dir}')
    print(f'[test] {len(inputs)} files from {test_dir}')

    output_root = output_zip.parent / 'tmp_infer_results'
    if output_root.exists():
        import shutil
        shutil.rmtree(output_root)
    for marker in MARKERS:
        (output_root / 'results' / 'test' / marker).mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    written = 0
    for off in range(0, len(inputs), batch_size):
        files = inputs[off:off + batch_size]
        arrays = [read_gray(f) for f in files]
        if any(a.shape != (256, 256) for a in arrays):
            raise ValueError('Expected 256x256 inputs')
        x = torch.from_numpy(np.stack(arrays)[:, None]).float().to(dev) / 255
        with amp_context(dev):
            pred = predict(model, x, tta)
            pred = jpeg_roundtrip(pred)
        pred_np = (pred.mul(255).round().to(torch.uint8).cpu().numpy()  # (B, 4, 256, 256)
                   [:, :, :, :])
        for file, channels in zip(files, pred_np):
            for marker, channel in zip(MARKERS, channels):
                rgb = np.repeat(channel[:, :, None], 3, axis=2)
                Image.fromarray(rgb).save(
                    output_root / 'results' / 'test' / marker /
                    (file.stem + '_fake.jpg'),
                    quality=OFFICIAL_JPEG_QUALITY, subsampling=0, optimize=False)
                written += 1
        if (off // batch_size) % 50 == 0:
            print(f'  {min(off + len(files), len(inputs))}/{len(inputs)}', flush=True)
    elapsed = time.perf_counter() - start
    print(f'[done] {written} JPEGs in {elapsed:.1f}s')

    # pack zip
    if output_zip.exists():
        output_zip.unlink()
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for marker in MARKERS:
            marker_dir = output_root / 'results' / 'test' / marker
            n = 0
            for f in sorted(marker_dir.glob('*_fake.jpg')):
                zf.write(f, f'results/test/{marker}/{f.name}')
                n += 1
            print(f'[zip] {marker}: {n}')
    print(f'[zip] {output_zip}  ({output_zip.stat().st_size/1024/1024:.1f} MB)')

    # save provenance
    prov = output_root / 'provenance.json'
    prov.write_text(json.dumps({
        'input_count': len(inputs),
        'markers': list(MARKERS),
        'tta': tta,
        'serialization': {'format': 'JPEG', 'quality': OFFICIAL_JPEG_QUALITY,
                          'subsampling': 0, 'optimize': False},
        'checkpoint_sha256': hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
        'checkpoint_name': ckpt_path.name,
    }, indent=2), encoding='utf-8')

    import shutil
    shutil.rmtree(output_root)
    print(f'[clean] removed {output_root}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--test-dir', default=r'E:\aic\ihc-data\test\DAPI')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tta', type=int, default=4, choices=(1, 4, 8))
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    infer_zip(Path(args.ckpt), Path(args.test_dir), Path(args.out),
              tta=args.tta, batch_size=args.batch_size, device=args.device)


if __name__ == '__main__':
    main()
