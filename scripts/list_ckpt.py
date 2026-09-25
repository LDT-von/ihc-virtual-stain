"""List all checkpoints."""
from pathlib import Path
root = Path(r'E:\aic\final-ihc')
for p in sorted(root.rglob('*.pt')) + sorted(root.rglob('*.pth')):
    print(f'{p.stat().st_size/1024/1024:6.1f} MB  {p}')
print('---')
for p in sorted(root.rglob('*v3*')):
    print(p)
