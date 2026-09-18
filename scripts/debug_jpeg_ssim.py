"""验证 JPEG 压缩质量差异是否导致 SSIM 极低"""
import sys
from pathlib import Path
sys.path.insert(0, r"E:\aic\ihc-virtual-stain")
from PIL import Image
import numpy as np
import torch
from src.metrics.ssim_psnr import ssim_gpu_single, psnr_gpu_single

ROOT = Path(r"E:\aic\ihc-virtual-stain\results\val\HLA-DR")
SRC = Path(r"E:\aic\ihc-data\val\HLA-DR")

# 选前 10 张做实验
names = sorted([p.stem for p in SRC.glob("*.jpg")])[:10]

for name in names:
    real_path = SRC / f"{name}.jpg"
    fake_path = ROOT / f"{name}_fake.jpg"

    # 1. 直接读两者算 SSIM
    real = np.array(Image.open(real_path).convert("RGB"))
    fake = np.array(Image.open(fake_path).convert("RGB"))

    # 2. 把 fake 用同样低质量保存再读
    tmp = ROOT / f"{name}_fake_recomp.jpg"
    Image.fromarray(fake).save(tmp, format="JPEG", quality=50)
    fake_recomp = np.array(Image.open(tmp).convert("RGB"))
    tmp.unlink()

    # 转 tensor
    def to_tensor(arr):
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return t * 2 - 1

    ssim_direct = ssim_gpu_single(to_tensor(fake).cuda(), to_tensor(real).cuda())
    ssim_recomp = ssim_gpu_single(to_tensor(fake_recomp).cuda(), to_tensor(real).cuda())

    psnr_direct = psnr_gpu_single(to_tensor(fake).cuda(), to_tensor(real).cuda())
    psnr_recomp = psnr_gpu_single(to_tensor(fake_recomp).cuda(), to_tensor(real).cuda())

    print(f"{name}: SSIM direct={ssim_direct:.4f} recomp={ssim_recomp:.4f} | "
          f"PSNR direct={psnr_direct:.2f} recomp={psnr_recomp:.2f}")
