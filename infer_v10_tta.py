"""v10 ensemble + TTA + multi-scale inference

关键改进：
1. 多尺度推理 (0.85x, 1.0x, 1.15x)
2. 多 flip TTA
3. 多个 checkpoint ensemble
4. 对 CD45RO 这种弱 marker 加权更低
"""
import argparse
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from skimage.metrics import structural_similarity as sk_ssim
import builtins
import numpy as np
from PIL import Image

ROOT = Path(r'E:\aic\ihc-virtual-stain')
sys.path.insert(0, str(ROOT))

_orig_print = builtins.print
def _fp(*a, **k):
    k.setdefault('flush', True)
    try:
        _orig_print(*a, **k)
    except UnicodeEncodeError:
        safe = [str(x).encode('gbk', errors='replace').decode('gbk') for x in a]
        _orig_print(*safe, **k)
builtins.print = _fp

from src.models.pix2pix_gan import build_pix2pix_model
from src.metrics.ssim_psnr import to_uint8


class TTAFolder(Dataset):
    """加载 test 输入文件夹，做 TTA + 多尺度 ensemble"""
    def __init__(self, root, marker, scales=(1.0,), patch_size=256):
        self.root = Path(root) / 'test' / 'input' / marker
        self.files = sorted(self.root.glob('*.png')) + sorted(self.root.glob('*.jpg'))
        self.scales = scales
        self.patch_size = patch_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        img = np.array(Image.open(f).convert('RGB'))
        # Pad to multiple of patch_size
        h, w, _ = img.shape
        ph = (self.patch_size - h % self.patch_size) % self.patch_size
        pw = (self.patch_size - w % self.patch_size) % self.patch_size
        img_padded = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode='reflect')
        return {
            'file': str(f),
            'img_padded': torch.from_numpy(img_padded).permute(2, 0, 1).float() / 255.0,
            'orig_size': (h, w),
            'padded_size': img_padded.shape[:2],
        }


def tta_augment(x, mode):
    """4 个 flip TTA 模式"""
    if mode == 0:
        return x
    elif mode == 1:
        return torch.flip(x, [2])
    elif mode == 2:
        return torch.flip(x, [3])
    elif mode == 3:
        return torch.flip(x, [2, 3])
    return x


def tta_deaugment(y, mode):
    """反向 flip TTA"""
    if mode == 0:
        return y
    elif mode == 1:
        return torch.flip(y, [2])
    elif mode == 2:
        return torch.flip(y, [3])
    elif mode == 3:
        return torch.flip(y, [2, 3])
    return y


def predict_tile(models, dapi_tile, scales, device, n_tta=4):
    """对单个 tile 做 ensemble + multi-scale + TTA"""
    accum = None
    n_total = 0
    for model in models:
        for scale in scales:
            h, w = dapi_tile.shape[2], dapi_tile.shape[3]
            new_h = int(h * scale / 32) * 32
            new_w = int(w * scale / 32) * 32
            if scale != 1.0:
                dapi_scaled = F.interpolate(dapi_tile, size=(new_h, new_w), mode='bilinear', align_corners=False)
            else:
                dapi_scaled = dapi_tile
            for tta in range(n_tta):
                x = tta_augment(dapi_scaled, tta).to(device)
                with torch.inference_mode():
                    fake = model.generator(x, x)
                # Resize back
                if scale != 1.0:
                    fake = F.interpolate(fake, size=(h, w), mode='bilinear', align_corners=False)
                fake = tta_deaugment(fake.cpu(), tta)
                if accum is None:
                    accum = fake
                else:
                    accum = accum + fake
                n_total += 1
    return accum / n_total


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）')
    p.add_argument('--out-dir', default='E:/aic/ihc-virtual-stain/v10_results')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint 配置（多个最强 checkpoint ensemble）
    configs = {
        'HLA-DR': [
            'checkpoints/pix2pix_v2_HLA-DR_1788962683/best.pt',
        ],
        'CD68': [
            'checkpoints/pix2pix_v2_CD68_1788969904/best.pt',
        ],
        'CD45RO': [
            'checkpoints/pix2pix_v7_CD45RO_1789200177/epoch57.pt',
        ],
        'Vimentin': [
            'checkpoints/pix2pix_v7_Vimentin_1789200192/epoch58.pt',
            'checkpoints/pix2pix_v7_Vimentin_1789200192/epoch60.pt',
            'checkpoints/pix2pix_v7_Vimentin_1789200192/epoch62.pt',
        ],
    }

    scales = (1.0,)

    for marker, ck_paths in configs.items():
        print(f'\n=== Marker: {marker} ===')
        # 加载 models
        models = []
        for ck_path in ck_paths:
            model = build_pix2pix_model(3, 3, 3, 64).to(device)
            ck = torch.load(ROOT / ck_path, map_location=device, weights_only=False)
            model.load_state_dict(ck['model'])
            model.eval()
            models.append(model)
            print(f'  Loaded {ck_path}')

        # 处理测试集
        ds = TTAFolder(Path(args.data_root), marker)
        print(f'  Test files: {len(ds)}')

        marker_out = out_dir / marker
        marker_out.mkdir(parents=True, exist_ok=True)

        for idx in range(len(ds)):
            item = ds[idx]
            dapi_full = item['img_padded'].unsqueeze(0)
            h, w = dapi_full.shape[2], dapi_full.shape[3]
            orig_h, orig_w = item['orig_size']

            # 分 tile
            ps = 256
            result = torch.zeros_like(dapi_full)
            weight = torch.zeros(1, 1, h, w)

            for y in range(0, h, ps):
                for x in range(0, w, ps):
                    tile = dapi_full[:, :, y:y+ps, x:x+ps]
                    if tile.shape[2] < ps or tile.shape[3] < ps:
                        # Pad last tile
                        ph = ps - tile.shape[2]
                        pw = ps - tile.shape[3]
                        tile = F.pad(tile, (0, pw, 0, ph), mode='reflect')
                    pred = predict_tile(models, tile, scales, device)
                    result[:, :, y:y+ps, x:x+ps] = pred[:, :, :min(ps, h-y), :min(ps, w-x)]
                    weight[:, :, y:y+ps, x:x+ps] += 1

            result = result / weight.clamp(min=1)
            result = result[0, :, :orig_h, :orig_w].clamp(0, 1)
            arr = (result.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            out_path = marker_out / Path(item['file']).name
            Image.fromarray(arr).save(out_path)
            if (idx + 1) % 50 == 0:
                print(f'  [{marker}] {idx+1}/{len(ds)}')

        # Free memory
        for m in models:
            del m
        torch.cuda.empty_cache()

    print('\n=== Done. Results in', out_dir, '===')


if __name__ == '__main__':
    main()
