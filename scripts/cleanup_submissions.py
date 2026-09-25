"""Delete invalid submission zips (L-mode PNGs and the L-mode JPG)."""
from pathlib import Path

sub_dir = Path(r'E:\aic\final-ihc\submissions')

# Keep: ensemble4_rgb, fullplus_cd68_v1_rgb, fullplus_cd68_v3_rgb_FIXED
# Delete: all invalid ones
to_delete = [
    'cd68_aux_v1.zip',
    'cd68_weighted_w96_v2.zip',
    'fullplus_cd68_v1.zip',
    'fullplus_cd68_v1_fixed.zip',
    'fullplus_cd68_v3.zip',
    'fullplus_cd68_v3_rgb.zip',
    'w128_expanded.zip',
]

freed_mb = 0
for name in to_delete:
    p = sub_dir / name
    if p.exists():
        sz = p.stat().st_size / 1024 / 1024
        p.unlink()
        freed_mb += sz
        print(f'  DELETED  {name:42s} ({sz:5.1f} MB)')
    else:
        print(f'  MISSING  {name}')

print(f'\nTotal freed: {freed_mb:.1f} MB')
print('\nRemaining:')
for p in sorted(sub_dir.glob('*.zip')):
    print(f'  {p.stat().st_size/1024/1024:6.1f} MB   {p.name}')
