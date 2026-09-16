"""对每个 marker 试多种 ensemble 组合，找最佳"""
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as sk_skim

ROOT = Path(r'E:\aic\ihc-virtual-stain')
sys.path.insert(0, str(ROOT))
from src.data.dataset import DAPItoIHCDataset
from src.metrics.ssim_psnr import to_uint8
from src.models.pix2pix_gan import build_pix2pix_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_root = Path(r'E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）')


def avg_sds(state_dicts):
    avg = {}
    for k in state_dicts[0].keys():
        if state_dicts[0][k].dtype.is_floating_point:
            avg[k] = sum(sd[k].float() for sd in state_dicts) / len(state_dicts)
        else:
            avg[k] = state_dicts[-1][k]
    return avg


def eval_model(model, marker, bs=8):
    ds = DAPItoIHCDataset(root=data_root, marker=marker, split='val',
                          patch_size=256, augment=False)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    ssim_sum = 0.0
    n = 0
    with torch.inference_mode():
        for batch in loader:
            dapi = batch['dapi'].to(device)
            real = batch['ihc'].to(device)
            fake = model.generator(dapi, dapi)
            for i in range(fake.shape[0]):
                p = to_uint8(fake[i].cpu())
                t = to_uint8(real[i].cpu())
                ssim_sum += sk_skim(p, t, channel_axis=-1, data_range=255)
            n += fake.shape[0]
    return ssim_sum / n


def test_combo(marker, ckpt_dir, epochs_list):
    cd = ROOT / 'checkpoints' / ckpt_dir
    sds = []
    for ep in epochs_list:
        ck = torch.load(cd / f'{ep}.pt', map_location='cpu', weights_only=False)
        sds.append(ck['model'])
    avg = avg_sds(sds)
    m = build_pix2pix_model(3, 3, 3, 64).to(device)
    m.load_state_dict(avg)
    m.eval()
    s = eval_model(m, marker)
    del m
    torch.cuda.empty_cache()
    return s


# CD45RO - test various combos
print('=== CD45RO Ensembles ===')
cd = 'pix2pix_v7_CD45RO_1789200177'
combos = [
    ['best'],
    ['epoch57'],
    ['epoch55', 'epoch57'],
    ['epoch56', 'epoch57', 'epoch58'],
    ['epoch57', 'epoch58', 'epoch59'],
    ['epoch55', 'epoch56', 'epoch57', 'epoch58', 'epoch59'],
    ['best', 'epoch57'],
    ['best', 'epoch55', 'epoch57'],
    ['best', 'epoch57', 'epoch59'],
    ['epoch53', 'epoch55', 'epoch57', 'epoch59'],
    ['epoch51', 'epoch53', 'epoch55', 'epoch57', 'epoch59'],
]
results = []
for c in combos:
    try:
        s = test_combo('CD45RO', cd, c)
        print(f'  {c}: {s:.4f}')
        results.append((c, s))
    except Exception as e:
        print(f'  {c}: ERROR {e}')

print('\n=== CD45RO sorted ===')
for c, s in sorted(results, key=lambda x: -x[1])[:5]:
    print(f'  {c}: {s:.4f}')

print('\n=== Vimentin Ensembles ===')
cd = 'pix2pix_v7_Vimentin_1789200192'
combos = [
    ['best'],
    ['epoch60'],
    ['epoch58', 'epoch60'],
    ['epoch60', 'epoch62'],
    ['epoch58', 'epoch60', 'epoch62'],
    ['epoch60', 'epoch62', 'epoch64'],
    ['best', 'epoch60'],
    ['best', 'epoch60', 'epoch62'],
    ['best', 'epoch60', 'epoch64'],
    ['epoch55', 'epoch60', 'epoch64'],
]
results = []
for c in combos:
    try:
        s = test_combo('Vimentin', cd, c)
        print(f'  {c}: {s:.4f}')
        results.append((c, s))
    except Exception as e:
        print(f'  {c}: ERROR {e}')

print('\n=== Vimentin sorted ===')
for c, s in sorted(results, key=lambda x: -x[1])[:5]:
    print(f'  {c}: {s:.4f}')
