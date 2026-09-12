"""v8 推理：使用最佳 epoch / ensemble

HLA-DR: best.pt (0.7700)
CD68: best.pt (0.7976)
CD45RO: epoch57 (0.7031)
Vimentin: epoch58+60+62 平均 (0.7515)
"""
import sys
import time
from pathlib import Path
from torch.utils.data import DataLoader
import torch

ROOT = Path(r'E:\aic\ihc-virtual-stain')
sys.path.insert(0, str(ROOT))
from src.data.dataset import DAPItoIHCDataset
from src.metrics.ssim_psnr import to_uint8
from src.models.pix2pix_gan import build_pix2pix_model


def avg_sds(state_dicts):
    avg = {}
    for k in state_dicts[0].keys():
        if state_dicts[0][k].dtype.is_floating_point:
            avg[k] = sum(sd[k].float() for sd in state_dicts) / len(state_dicts)
        else:
            avg[k] = state_dicts[-1][k]
    return avg


def get_model(marker):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck_root = ROOT / 'checkpoints'
    if marker == 'HLA-DR':
        ck = torch.load(ck_root / 'pix2pix_v2_HLA-DR_1788962683' / 'best.pt',
                       map_location=device, weights_only=False)
    elif marker == 'CD68':
        ck = torch.load(ck_root / 'pix2pix_v2_CD68_1788969904' / 'best.pt',
                       map_location=device, weights_only=False)
    elif marker == 'CD45RO':
        ck = torch.load(ck_root / 'pix2pix_v7_CD45RO_1789200177' / 'epoch57.pt',
                       map_location=device, weights_only=False)
    elif marker == 'Vimentin':
        sds = []
        for ep in ['epoch58', 'epoch60', 'epoch62']:
            ck = torch.load(ck_root / 'pix2pix_v7_Vimentin_1789200192' / f'{ep}.pt',
                          map_location='cpu', weights_only=False)
            sds.append(ck['model'])
        avg = avg_sds(sds)
        m = build_pix2pix_model(3, 3, 3, 64).to(device)
        m.load_state_dict(avg)
        return m
    model = build_pix2pix_model(3, 3, 3, 64).to(device)
    model.load_state_dict(ck['model'])
    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = Path(r'E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）')
    output_root = ROOT / 'results_v8_test' / 'test'

    for marker in ['HLA-DR', 'CD68', 'CD45RO', 'Vimentin']:
        print(f'\n[{marker}]')
        model = get_model(marker)
        model.eval()

        ds = DAPItoIHCDataset(root=data_root, marker=marker, split='test',
                              patch_size=256, augment=False)
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

        output_dir = output_root / marker
        output_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        with torch.inference_mode():
            for batch in loader:
                dapi = batch['dapi'].to(device)
                fake = model.generator(dapi, dapi)
                for i, name in enumerate(batch['name']):
                    from PIL import Image
                    img = Image.fromarray(to_uint8(fake[i].cpu()))
                    img.save(output_dir / f'{name}_fake.jpg', format='JPEG', quality=95)

        print(f'  [{marker}] done in {time.time()-t0:.1f}s')
        del model
        torch.cuda.empty_cache()

    print('\n[Done] results in', output_root)


if __name__ == '__main__':
    main()
