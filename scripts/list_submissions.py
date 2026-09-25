"""List submission zip files with sizes."""
from pathlib import Path
sub_dir = Path(r'E:\aic\final-ihc\submissions')
for p in sorted(sub_dir.glob('*.zip')):
    print(f'  {p.stat().st_size/1024/1024:6.1f} MB   {p.name}')
