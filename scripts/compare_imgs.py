"""Compare image contents between successful and failed submissions."""
import zipfile
import numpy as np
import io
from PIL import Image

def load_img_from_zip(zp, name):
    with zipfile.ZipFile(zp) as z:
        with z.open(name) as f:
            return np.array(Image.open(io.BytesIO(f.read())))

# Compare same ROI across submissions
samples = [
    ('results/test/HLA-DR/ROI017_00_00_fake.jpg', 'good'),
    ('results/test/HLA-DR/ROI017_00_00_HLA-DR.png', 'v3'),
    ('results/test/CD68/ROI025_00_05_fake.jpg', 'good'),
    ('results/test/CD68/ROI025_00_05_CD68.png', 'v3'),
]

good_zip = r'E:\aic\final-ihc\submission_w96_expanded_tta4_semi.zip'
v3_zip = r'E:\aic\final-ihc\submissions\fullplus_cd68_v3.zip'

for name, kind in samples:
    zp = good_zip if kind == 'good' else v3_zip
    try:
        arr = load_img_from_zip(zp, name)
        print(f'{name[:50]:50} | shape={arr.shape} dtype={arr.dtype} min={arr.min()} max={arr.max()} mean={arr.mean():.2f}')
    except Exception as e:
        print(f'{name[:50]:50} | ERROR: {e}')
