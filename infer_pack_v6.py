# -*- coding: utf-8
"""Infer v6 final (best snapshot) and pack submission zip."""
import subprocess, sys
ROOT = r'E:\aic\final-ihc'

# infer v6 final.pt
cmd = [
    sys.executable, '-X', 'utf8', '-u', '-m', 'src.train_marker_context', 'infer',
    '--output', r'E:\aic\final-ihc\results\infer_v6',
    '--device', 'cuda',
    '--batch-size', '8',
    '--tta', '4',
    '--checkpoint', r'E:\aic\final-ihc\checkpoints\fullplus_cd68_v6\final.pt',
    '--input', r'E:\aic\final-ihc\test_semi_DAPI',
    '--seed', '2026',
]
print('infer v6 final.pt')
ret = subprocess.run(cmd, cwd=ROOT)
if ret.returncode != 0:
    print('INFER FAILED')
    sys.exit(1)

# Pack
import zipfile, hashlib, os
from pathlib import Path
SUB = Path(ROOT) / 'submissions'
SUB.mkdir(parents=True, exist_ok=True)
out_zip = SUB / 'fullplus_cd68_v6_rgb.zip'
MARKERS = ['HLA-DR', 'CD68', 'CD45RO', 'Vimentin']
total = 0
matched = {m: 0 for m in MARKERS}
results_dir = Path(ROOT) / 'results' / 'infer_v6' / 'results' / 'test'
if not results_dir.exists():
    print('unexpected layout')
    sys.exit(1)
with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for marker in MARKERS:
        md = results_dir / marker
        if not md.exists():
            print(f'WARN: {md} missing')
            continue
        for j in sorted(md.glob('*.jpg')):
            arcname = f'results/test/{marker}/{j.name}'
            zf.write(j, arcname)
            total += 1
            matched[marker] += 1
sz = out_zip.stat().st_size / 1e6
sha = hashlib.sha256(open(out_zip, 'rb').read()).hexdigest()[:16]
print(f'\npacked: {out_zip.name} ({sz:.1f}MB, sha={sha})')
print(f'  per-marker: {matched}')
