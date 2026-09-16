"""对 CD45RO 用最近 5 个 epoch 做模型 ensemble（权重平均）

如果发现提升，再应用到其他 marker。
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as sk_ssim

ROOT = Path(r'E:\aic\ihc-virtual-stain')
sys.path.insert(0, str(ROOT))
from src.data.dataset import DAPItoIHCDataset
from src.metrics.ssim_psnr import to_uint8
from src.models.pix2pix_gan import build_pix2pix_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_root = Path(r'E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）')


def avg_state_dicts(state_dicts):
    """平均多个 state_dict"""
    avg = {}
    for k in state_dicts[0].keys():
        if state_dicts[0][k].dtype.is_floating_point:
            avg[k] = sum(sd[k].float() for sd in state_dicts) / len(state_dicts)
        else:
            # 整数缓冲（如 num_batches_tracked）取最后一个
            avg[k] = state_dicts[-1][k]
    return avg


def evaluate_with_state(model, marker, bs=8):
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
                ssim_sum += sk_ssim(p, t, channel_axis=-1, data_range=255)
            n += fake.shape[0]
    return ssim_sum / n


def ensemble_eval(marker, ckpt_dir, epoch_names):
    cd = ROOT / 'checkpoints' / ckpt_dir
    state_dicts = []
    for name in epoch_names:
        ck_path = cd / (f'{name}.pt' if name != 'best' else 'best.pt')
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        state_dicts.append(ck['model'])

    avg_sd = avg_state_dicts(state_dicts)

    model = build_pix2pix_model(3, 3, 3, 64).to(device)
    model.load_state_dict(avg_sd)
    model.eval()

    s = evaluate_with_state(model, marker)
    print(f'  Ensemble {epoch_names}: skimage SSIM = {s:.4f}')
    del model
    torch.cuda.empty_cache()
    return s


if __name__ == '__main__':
    # CD45RO: best + epoch57~59
    print('\n=== CD45RO Ensemble ===')
    e_best = ensemble_eval('CD45RO', 'pix2pix_v7_CD45RO_1789200177', ['best'])
    e_57 = ensemble_eval('CD45RO', 'pix2pix_v7_CD45RO_1789200177', ['epoch57'])
    e_55_59 = ensemble_eval('CD45RO', 'pix2pix_v7_CD45RO_1789200177', ['epoch55', 'epoch56', 'epoch57', 'epoch58', 'epoch59'])
    e_50_59 = ensemble_eval('CD45RO', 'pix2pix_v7_CD45RO_1789200177', ['epoch50', 'epoch52', 'epoch54', 'epoch57', 'epoch59'])
    e_all_late = ensemble_eval('CD45RO', 'pix2pix_v7_CD45RO_1789200177', ['epoch40', 'epoch45', 'epoch50', 'epoch55', 'epoch57'])
