"""Audit each submission zip: format compliance + file count."""
import zipfile, io
from pathlib import Path
from PIL import Image

sub_dir = Path(r'E:\aic\final-ihc\submissions')
for zp in sorted(sub_dir.glob('*.zip')):
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        n = len(names)
        # Filename check
        bad_name = sum(1 for x in names if not x.endswith('_fake.jpg'))
        # Extension check
        bad_ext = sum(1 for x in names if not x.endswith('.jpg'))
        # Mode check (sample first)
        sample = names[0]
        data = z.read(sample)
        im = Image.open(io.BytesIO(data))
        mode = im.mode
        size = im.size
    print(f'{zp.name:42s} | {n:4d} files | mode={mode} {size} | bad_name={bad_name} bad_ext={bad_ext}')
