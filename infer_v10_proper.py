"""v10 inference：使用官方 DAPItoIHCDataset (split='test') + TTA + ensemble

关键改进：
1. 4-flip TTA
2. Vimentin 3-checkpoint ensemble
3. 复用平台兼容的数据加载方式
"""
import sys
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

ROOT = Path(r'E:\aic\ihc-virtual-stain')
sys.path.insert(0, str(ROOT))

from src.data.dataset import DAPItoIHCDataset
from src.metrics.ssim_psnr import to_uint8
from src.models.pix2pix_gan import build_pix2pix_model
from PIL import Image


def avg_sds(state_dicts):
    avg = {}
    for k in state_dicts[0].keys():
        if state_dicts[0][k].dtype.is_floating_point:
            avg[k] = sum(sd[k].float() for sd in state_dicts) / len(state_dicts)
        else:
            avg[k] = state_dicts[-1][k]
    return avg


def get_models(marker, device):
    """每个 marker 返回 1 个或多个模型"""
    ck_root = ROOT / 'checkpoints'
    if marker == 'HLA-DR':
        ck = torch.load(ck_root / 'pix2pix_v2_HLA-DR_1788962683' / 'best.pt',
                       map_location=device, weights_only=False)
        m = build_pix2pix_model(3, 3, 3, 64).to(device)
        m.load_state_dict(ck['model'])
        return [m]
    elif marker == 'CD68':
        ck = torch.load(ck_root / 'pix2pix_v2_CD68_1788969904' / 'best.pt',
                       map_location=device, weights_only=False)
        m = build_pix2pix_model(3, 3, 3, 64).to(device)
        m.load_state_dict(ck['model'])
        return [m]
    elif marker == 'CD45RO':
        ck = torch.load(ck_root / 'pix2pix_v7_CD45RO_1789200177' / 'epoch57.pt',
                       map_location=device, weights_only=False)
        m = build_pix2pix_model(3, 3, 3, 64).to(device)
        m.load_state_dict(ck['model'])
        return [m]
    elif marker == 'Vimentin':
        sds = []
        for ep in ['epoch58', 'epoch60', 'epoch62']:
            ck = torch.load(ck_root / 'pix2pix_v7_Vimentin_1789200192' / f'{ep}.pt',
                          map_location='cpu', weights_only=False)
            sds.append(ck['model'])
        avg = avg_sds(sds)
        m = build_pix2pix_model(3, 3, 3, 64).to(device)
        m.load_state_dict(avg)
        return [m]


def tta_flip(x, mode):
    if mode == 0: return x
    elif mode == 1: return torch.flip(x, [3])  # horizontal
    elif mode == 2: return torch.flip(x, [2])  # vertical
    elif mode == 3: return torch.flip(x, [2, 3])  # both
    return x


def tta_deflip(y, mode):
    if mode == 0: return y
    elif mode == 1: return torch.flip(y, [3])
    elif mode == 2: return torch.flip(y, [2])
    elif mode == 3: return torch.flip(y, [2, 3])
    return y


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = Path(r'E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）')
    output_root = ROOT / 'results_v10_test' / 'test'
    n_tta = 4

    for marker in ['HLA-DR', 'CD68', 'CD45RO', 'Vimentin']:
        print(f'\n[{marker}]')
        models = get_models(marker, device)
        for m in models:
            m.eval()
        print(f'  {len(models)} model(s), {n_tta}-flip TTA')

        ds = DAPItoIHCDataset(root=data_root, marker=marker, split='test',
                              patch_size=256, augment=False)
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        print(f'  Test samples: {len(ds)}')

        output_dir = output_root / marker
        output_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        with torch.inference_mode():
            for bi, batch in enumerate(loader):
                dapi = batch['dapi'].to(device)

                # TTA + ensemble
                accum = None
                n_total = 0
                for model in models:
                    for tta in range(n_tta):
                        x = tta_flip(dapi, tta)
                        fake = model.generator(x, x)
                        fake = tta_deflip(fake, tta)
                        if accum is None:
                            accum = fake
                        else:
                            accum = accum + fake
                        n_total += 1
                pred = accum / n_total

                for i, name in enumerate(batch['name']):
                    img = Image.fromarray(to_uint8(pred[i].cpu()))
                    img.save(output_dir / f'{name}_fake.jpg', format='JPEG', quality=95)

                if (bi + 1) % 20 == 0:
                    print(f'  [{marker}] {bi+1}/{len(loader)}')

        print(f'  [{marker}] done in {time.time()-t0:.1f}s')
        for m in models:
            del m
        torch.cuda.empty_cache()

    print('\n[Done] results in', output_root)


if __name__ == '__main__':
    main()
