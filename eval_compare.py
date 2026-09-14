import torch, sys, random, numpy as np
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, r'e:\aic\ihc-virtual-stain')
from src.data.dataset import DAPItoIHCDataset
from src.models.pix2pix_gan import build_pix2pix_model
from src.metrics.ssim_psnr import ssim as skimage_ssim

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
                ssims.append(skimimage_ssim(fake[i], ihc[i]))
    return np.mean(ssims)

v11_dir = ckpt_base + r'\pix2pix_v11_HLA-DR_1789355023'
v2_map = {
    'CD68': 'pix2pix_v2_CD68_1788969904',
    'HLA-DR': 'pix2pix_v2_HLA-DR_1788962683',
    'CD45RO': 'pix2pix_v2_CD45RO_1788978395',
    'Vimentin': 'pix2pix_v2_Vimentin_1788988426',
}
val_n = 250
for marker in ['CD68', 'HLA-DR', 'CD45RO', 'Vimentin']:
    full_ds = DAPItoIHCDataset(root=data_root, marker=marker, split='train', patch_size=256, augment=False)
    rng = random.Random(42)
    val_idx = sorted(rng.sample(range(len(full_ds)), val_n))
    val_ds = Subset(full_ds, val_idx)
    loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    
    v2_ckpt = ckpt_base + r'\\' + v2_map[marker] + r'\\best.pt'
    v2_ssim = eval_model(v2_ckpt, loader)
    
    if marker == 'HLA-DR':
        v11_ssim = eval_model(v11_dir + r'\\best.pt', loader)
        print(f'{marker}: v11={v11_ssim:.4f}  v2={v2_ssim:.4f}  diff={v11_ssim-v2_ssim:+.4f}')
    else:
        print(f'{marker}: v2={v2_ssim:.4f} (v11 not yet trained)')
