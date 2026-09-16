import torch, sys, random, numpy as np
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, r'e:\aic\ihc-virtual-stain')
from src.data.dataset import DAPItoIHCDataset
from src.models.pix2pix_gan import build_pix2pix_model
from src.metrics.ssim_psnr import ssim as _ssim

device = torch.device('cuda')
data_root = r'E:\AIC\ihc-virtual-stain\初赛数据集（包含训练集和测试集输入）\初赛数据集（包含训练集和测试集输入）'
ckpt_base = r'e:\aic\ihc-virtual-stain\checkpoints'

def eval_model(ckpt_path, loader):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_pix2pix_model(3,3,3,64).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    ssims = []
    with torch.inference_mode():
        for batch in loader:
            dapi = batch['dapi'].to(device)
            ihc = batch['ihc'].to(device)
            fake = model.generator(dapi, dapi)
            for i in range(fake.shape[0]):
                ssims.append(_ssim(fake[i], ihc[i]))
    return np.mean(ssims)

val_n = 250
full_ds = DAPItoIHCDataset(root=data_root, marker='HLA-DR', split='train', patch_size=256, augment=False)
rng = random.Random(42)
val_idx = sorted(rng.sample(range(len(full_ds)), val_n))
val_ds = Subset(full_ds, val_idx)
loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

v11_ckpt = ckpt_base + '\\pix2pix_v11_HLA-DR_1789361914\\best.pt'
v2_ckpt = ckpt_base + '\\pix2pix_v2_HLA-DR_1788962683\\best.pt'

v11_ssim = eval_model(v11_ckpt, loader)
v2_ssim = eval_model(v2_ckpt, loader)
print(f'v11: {v11_ssim:.4f}')
print(f'v2:  {v2_ssim:.4f}')
print(f'gap:  {v11_ssim-v2_ssim:+.4f}')
print(f'v11 platform score: {v11_ssim*100:.2f}')
