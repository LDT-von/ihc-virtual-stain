"""检查是否目录角色交换了 - 比较 train/DAPI 和 val/IHC 的像素相关性"""
import numpy as np
from PIL import Image
from pathlib import Path

# 假设: train/HLA-DR 里的空白图实际上是真正的 DAPI 核染色
# 假设: val/DAPI 的空白图实际是 IHC（但 val/HLA-DR 更大）

# 方案: 把 train/DAPI 复制到 ihc-data3/train2/IHC_HLA-DR
# 把 train/HLA-DR 复制到 ihc-data3/train2/DAPI
# 然后用 train2 训练，再在 val 上评估

import shutil
from pathlib import Path

src_train = Path(r'E:\aic\ihc-data\train')
dst = Path(r'E:\aic\ihc-data3\train2')
dst.mkdir(parents=True, exist_ok=True)

# 交换
src_dapi = src_train / 'DAPI'
src_ihc = src_train / 'HLA-DR'
dst_dapi = dst / 'DAPI'
dst_ihc = dst / 'IHC_HLA-DR'

if dst_dapi.exists():
    shutil.rmtree(dst_dapi)
if dst_ihc.exists():
    shutil.rmtree(dst_ihc)

import subprocess
subprocess.run(['cmd', '/c', 'mklink', '/J', str(dst_dapi), str(src_ihc)], check=True)  # DAPI <- 空白目录
subprocess.run(['cmd', '/c', 'mklink', '/J', str(dst_ihc), str(src_dapi)], check=True)  # IHC <- 真图目录

print("Swapped: DAPI now points to HLA-DR (blank), IHC points to DAPI (real)")

# 验证
from PIL import Image
import numpy as np

dapi_file = sorted(Path(r'E:\aic\ihc-data3\train2\DAPI').glob('*.jpg'))[0]
ihc_file = sorted(Path(r'E:\aic\ihc-data3\train2\IHC_HLA-DR').glob('*.jpg'))[0]

d = np.array(Image.open(dapi_file).convert('L'))
i = np.array(Image.open(ihc_file).convert('L'))
print(f"train2/DAPI: mean={d.mean():.1f} std={d.std():.1f}")
print(f"train2/IHC_HLA-DR: mean={i.mean():.1f} std={i.std():.1f}")
