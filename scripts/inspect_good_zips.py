"""Inspect successful submission format."""
import zipfile
from pathlib import Path

zips = [
    r'E:\aic\final-ihc\submission_w96_expanded_tta4_semi.zip',
    r'E:\aic\final-ihc\75.0876\submission\submission_w96_expanded_tta4_semi.zip',
    r'E:\aic\final-ihc\predictions\ultimate_w48_2380_rgb\submission.zip',
]

for zp in zips:
    if not Path(zp).exists():
        print(f'\n=== {zp} === NOT FOUND')
        continue
    print(f'\n=== {zp} ===')
    print(f'  size: {Path(zp).stat().st_size/1024/1024:.1f} MB')
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        print(f'  files: {len(names)}')
        print('  first 6:')
        for n in names[:6]:
            print(f'    {n}')
        # unique ROIs
        rois = set()
        for n in names:
            parts = n.split('/')
            for p in parts:
                if p.startswith('ROI') and p.split('_')[0].startswith('ROI'):
                    rois.add(p.split('_')[0])
                    break
        print(f'  unique ROIs (sample): {sorted(rois)[:8]} ... {sorted(rois)[-4:]} (total {len(rois)})')
