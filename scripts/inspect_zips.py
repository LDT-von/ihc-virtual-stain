"""Inspect submission zips."""
import sys
import zipfile
from pathlib import Path

zips = [
    r'E:\aic\final-ihc\submissions\fullplus_cd68_v3.zip',
    r'E:\aic\final-ihc\submissions\fullplus_cd68_v3_rgb.zip',
]

for zp in zips:
    print(f'\n=== {zp} ===')
    print(f'  size: {Path(zp).stat().st_size/1024/1024:.1f} MB')
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        print(f'  files: {len(names)}')
        print('  first 8 entries:')
        for n in names[:8]:
            print(f'    {n}')
        print('  last 4 entries:')
        for n in names[-4:]:
            print(f'    {n}')
        # count by directory level
        levels = {}
        for n in names:
            depth = n.count('/')
            levels[depth] = levels.get(depth, 0) + 1
        print(f'  by depth: {levels}')
