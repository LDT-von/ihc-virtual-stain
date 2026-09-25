"""Pack v3 PNG outputs into official-format submission zip.

Format required (per SEMIFINAL_COMPLIANCE):
  - path: results/test/<MARKER>/<base>_fake.jpg
  - 256x256, 8-bit RGB JPEG, quality=100, subsampling=0, optimize=false
"""
import zipfile, io, os
from pathlib import Path
from PIL import Image

MARKERS = ['HLA-DR', 'CD68', 'CD45RO', 'Vimentin']
src_dir = Path(r'E:\aic\final-ihc\results_v3\test')
out_zip = Path(r'E:\aic\final-ihc\submissions\fullplus_cd68_v3_rgb_FIXED.zip')

print(f"Source: {src_dir}")
print(f"Output: {out_zip}")

total = 0
total_bytes = 0
with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for marker in MARKERS:
        marker_dir = src_dir / marker
        pngs = sorted(marker_dir.glob('*.png'))
        print(f"  {marker}: {len(pngs)} PNGs")
        for png_path in pngs:
            base = png_path.stem  # ROI017_00_00_HLA-DR
            for m in MARKERS:
                base = base.replace(f'_{m}', '')
            arcname = f'results/test/{marker}/{base}_fake.jpg'

            with Image.open(png_path) as im:
                # L -> RGB
                rgb = Image.new('RGB', im.size)
                rgb.paste(im, (0, 0))
                buf = io.BytesIO()
                rgb.save(buf, format='JPEG', quality=100, subsampling=0, optimize=False)
                buf.seek(0)
                data = buf.read()
                total_bytes += len(data)
                zf.writestr(arcname, data)
            total += 1

size_mb = os.path.getsize(out_zip) / 1e6
print(f"\nDone: {total} files -> {out_zip} ({size_mb:.1f} MB)")
print(f"Avg file: {total_bytes/total/1024:.1f} KB")

# Verify - load one JPG back, check mode
print("\nVerifying...")
with zipfile.ZipFile(out_zip) as zf:
    name = zf.namelist()[0]
    data = zf.read(name)
    im = Image.open(io.BytesIO(data))
    print(f"  {name}: mode={im.mode} size={im.size}")
    # Check name format
    print(f"  Total files: {len(zf.namelist())}")
    # Sample names
    print(f"  First 3: {zf.namelist()[:3]}")
