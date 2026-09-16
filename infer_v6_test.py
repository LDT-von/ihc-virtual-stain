"""用 v6 best checkpoint 直接跑 test 推理（无 TTA），打包提交
v6 各 marker val SSIM:
  CD68:    0.9276
  HLA-DR:  0.9100
  Vimentin: 0.8820
  CD45RO:  0.8812
"""
import os
import sys
import time
import zipfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

ROOT = Path(r'E:\aic\ihc-virtual-stain')
CKPT_BASE = ROOT / 'checkpoints'
OUTPUT_ROOT = ROOT / 'results_v6_test'
SUB_NAME = 'submission_v13_v6best.zip'

DATA_ROOT = Path(r'E:\AIC\ihc-virtual-stain\初赛数据集（包含训练集和测试集输入）\初赛数据集（包含训练集和测试集输入）')

# v6 best checkpoints
BEST_CKPTS = {
    'CD68':      CKPT_BASE / 'pix2pix_v6_CD68_1789186851' / 'best.pt',
    'HLA-DR':    CKPT_BASE / 'pix2pix_v6_HLA-DR_1789185490' / 'best.pt',
    'Vimentin':  CKPT_BASE / 'pix2pix_v6_Vimentin_1789189641' / 'best.pt',
    'CD45RO':    CKPT_BASE / 'pix2pix_v6_CD45RO_1789188139' / 'best.pt',
}

# 确保输出目录存在
for marker in BEST_CKPTS:
    (OUTPUT_ROOT / 'test' / marker).mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------
# 1. 确认 checkpoint 存在
# -------------------------------------------------------
print('=' * 60)
print('v6 Test Inference (No TTA)')
print('=' * 60)
all_ok = True
for marker, ckpt in BEST_CKPTS.items():
    if ckpt.exists():
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        ssim = ck.get('val_ssim', '?')
        ep = ck.get('epoch', '?')
        print(f'[OK] {marker}: val_ssim={ssim:.4f} epoch={ep}  ({ckpt.name})')
    else:
        print(f'[MISS] {marker}: {ckpt} NOT FOUND')
        all_ok = False

if not all_ok:
    sys.exit(1)

# -------------------------------------------------------
# 2. 对每个 marker 跑推理
# -------------------------------------------------------
sys.path.insert(0, str(ROOT))
from src.data.dataset import DAPItoIHCDataset
from src.models.pix2pix_gan import build_pix2pix_model
from src.metrics.ssim_psnr import MetricAggregator, to_uint8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice: {device}\n')

results = {}

for marker, ckpt_path in BEST_CKPTS.items():
    print(f'\n--- {marker} ---')
    t0 = time.time()

    # 加载模型
    model = build_pix2pix_model(
        input_channels=3, cond_channels=3, output_channels=3, base_filters=64
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # 测试集
    test_ds = DAPItoIHCDataset(
        root=DATA_ROOT, marker=marker, split='test',
        patch_size=256, augment=False,
    )
    loader = DataLoader(test_ds, batch_size=8, shuffle=False,
                       num_workers=0, pin_memory=True)

    print(f'Dataset: {len(test_ds)} patches')

    metric = MetricAggregator()
    written = 0

    with torch.inference_mode():
        for batch in loader:
            dapi = batch['dapi'].to(device, non_blocking=True)
            fake = model.generator(dapi, dapi)   # 无 TTA

            metric.update(fake, batch['ihc'].to(device))

            for i, name in enumerate(batch['name']):
                from PIL import Image
                img = Image.fromarray(to_uint8(fake[i].cpu()))
                img.save(OUTPUT_ROOT / 'test' / marker / f'{name}_fake.jpg',
                         format='JPEG', quality=95)
                written += 1

    r = metric.result()
    elapsed = time.time() - t0
    print(f'{marker}: SSIM={r["ssim"]:.4f} PSNR={r["psnr"]:.2f}  '
          f'{written} imgs  {elapsed:.1f}s')
    results[marker] = r

# -------------------------------------------------------
# 3. 汇总
# -------------------------------------------------------
print('\n' + '=' * 60)
print('v6 Test 结果汇总（无 TTA）:')
total_ssim = 0.0
for marker, r in results.items():
    print(f'  {marker}: SSIM={r["ssim"]:.4f} PSNR={r["psnr"]:.2f}')
    total_ssim += r['ssim']
avg = total_ssim / len(results)
print(f'\n  平均 SSIM: {avg:.4f}')
print(f'  平台预估分: {avg * 100:.2f}')
print(f'  输出目录: {OUTPUT_ROOT}')
print('=' * 60)

# -------------------------------------------------------
# 4. 打包提交
# -------------------------------------------------------
print(f'\n正在打包为 {SUB_NAME} ...')
sub_path = OUTPUT_ROOT / SUB_NAME
test_dir = OUTPUT_ROOT / 'test'

# 构建 zip，结构: test/HLA-DR/xxx_fake.jpg ...
with zipfile.ZipFile(sub_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for marker in BEST_CKPTS:
        marker_dir = test_dir / marker
        if not marker_dir.exists():
            print(f'[WARN] {marker_dir} 不存在，跳过')
            continue
        files = sorted(marker_dir.glob('*_fake.jpg'))
        for f in files:
            arcname = f'test/{marker}/{f.name}'
            zf.write(f, arcname)
        print(f'  {marker}: {len(files)} files')

size_mb = sub_path.stat().st_size / 1024 / 1024
print(f'打包完成: {sub_path}  ({size_mb:.1f} MB)')
print('\n请手动上传到评测平台！')
