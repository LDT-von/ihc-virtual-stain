"""Pack v3 PNGs into RGB JPG submission zip."""
import zipfile, io
from pathlib import Path
from PIL import Image

MARKERS = ['HLA-DR', 'CD68', 'CD45RO', 'Vimentin']
src_dir = Path(r'E:\aic\final-ihc\results_v3\test')
out_zip = r'E:\aic\final-ihc\submissions\fullplus_cd68_v3_rgb.zip'

print(f"Creating {out_zip}")
total = 0
with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for marker in MARKERS:
        marker_dir = src_dir / marker
        pngs = sorted(marker_dir.glob('*.png'))
        print(f"{marker}: {len(pngs)} PNGs")
        for png_path in pngs:
            # Strip marker suffix
            base = png_path.stem
            for m in MARKERS:
                base = base.replace(f'_{m}', '')
            arcname = f'results/test/{marker}/{base}_fake.jpg'

            with Image.open(png_path) as im:
                rgb = Image.new('RGB', im.size)
                rgb.paste(im, (0, 0))
                buf = io.BytesIO()
                rgb.save(buf, format='JPEG', quality=95, optimize=False)
                buf.seek(0)
                zf.writestr(arcname, buf.read())
            total += 1

import os
size_mb = os.path.getsize(out_zip) / 1e6
print(f"Done: {total} files -> {out_zip} ({size_mb:.1f} MB)")

# Verify
print("\nVerifying RGB mode...")
with zipfile.ZipFile(out_zip) as zf:
    for name in zf.namelist()[:3]:
        data = zf.read(name)
        im = Image.open(io.BytesIO(data))
        print(f"  {name}: mode={im.mode} size={im.size}")
