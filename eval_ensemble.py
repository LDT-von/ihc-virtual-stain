"""
Ensemble 评估：v2 + v6 融合，看是否能超越纯 v2

策略：
- v2: val SSIM 高（强 baseline）
- v6: val SSIM 低但预测可能更锐利
- ensemble: 加权平均（0.7*v2 + 0.3*v6 或搜索最优权重）
"""
import torch, sys, random, numpy as np
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, r'e:\aic\ihc-virtual-stain')
from src.data.dataset import DAPItoIHCDataset
from src.models.pix2pix_gan import build_pix2pix_model
from src.metrics.ssim_psnr import ssim as _ssim

device = torch.device('cuda')
data_root = r'E:\AIC\ihc-virtual-stain\初赛数据集（包含训练集和测试集输入）\初赛数据集（包含训练集和测试集输入）'
ckpt_base = r'e:\aic\ihc-virtual-stain\checkpoints'

def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = build_pix2pix_model(3,3,3,64).to(device)
    m.load_state_dict(ckpt['model'])
    m.eval()
    return m

def eval_ensemble(v2_ckpt, v6_ckpt, loader, w_v2):
    m2 = load_model(v2_ckpt)
    m6 = load_model(v6_ckpt)
    ssims_v2, ssims_v6, ssims_ens = [], [], []
    
    with torch.inference_mode():
        for batch in loader:
            dapi = batch['dapi'].to(device)
            ihc = batch['ihc'].to(device)
            
            f2 = m2.generator(dapi, dapi)
            f6 = m6.generator(dapi, dapi)
            f_ens = w_v2 * f2 + (1 - w_v2) * f6
            
            for i in range(dapi.shape[0]):
                ss2 = _ssim(f2[i], ihc[i])
                ss6 = _ssim(f6[i], ihc[i])
                ss_ens = _ssim(f_ens[i], ihc[i])
                ssims_v2.append(ss2)
                ssims_v6.append(ss6)
                ssims_ens.append(ss_ens)
    
    return np.mean(ssims_v2), np.mean(ssims_v6), np.mean(ssims_ens)

# v2 and v6 checkpoints
checkpoints = {
    'CD68': {
        'v2': ckpt_base + '\\pix2pix_v2_CD68_1788969904\\best.pt',
        'v6': ckpt_base + '\\pix2pix_v6_CD68_1789186851\\best.pt',
    },
    'HLA-DR': {
        'v2': ckpt_base + '\\pix2pix_v2_HLA-DR_1788962683\\best.pt',
        'v6': ckpt_base + '\\pix2pix_v6_HLA-DR_1789185490\\best.pt',
    },
    'CD45RO': {
        'v2': ckpt_base + '\\pix2pix_v2_CD45RO_1788978395\\best.pt',
        'v6': ckpt_base + '\\pix2pix_v6_CD45RO_1789188139\\best.pt',
    },
    'Vimentin': {
        'v2': ckpt_base + '\\pix2pix_v2_Vimentin_1788988426\\best.pt',
        'v6': ckpt_base + '\\pix2pix_v6_Vimentin_1789189641\\best.pt',
    },
}

val_n = 250
results = {}

for marker, cpts in checkpoints.items():
    print(f'\n=== {marker} ===')
    ds = DAPItoIHCDataset(root=data_root, marker=marker, split='train', patch_size=256, augment=False)
    rng = random.Random(42)
    val_idx = sorted(rng.sample(range(len(ds)), val_n))
    loader = DataLoader(Subset(ds, val_idx), batch_size=8, shuffle=False, num_workers=0)
    
    # 尝试不同权重
    best_w, best_ssim = 1.0, 0.0
    for w in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        ss2, ss6, ss_ens = eval_ensemble(cpts['v2'], cpts['v6'], loader, w)
        print(f'  w={w:.1f}: v2={ss2:.4f} v6={ss6:.4f} ens={ss_ens:.4f}')
        if ss_ens > best_ssim:
            best_ssim = ss_ens
            best_w = w
    
    results[marker] = {'best_w': best_w, 'best_ssim': best_ssim}
    print(f'  BEST: w={best_w:.1f} ssim={best_ssim:.4f}')

# 汇总
print('\n=== SUMMARY ===')
total = 0
for marker, r in results.items():
    print(f'{marker}: w={r["best_w"]:.1f} ssim={r["best_ssim"]:.4f}')
    total += r['best_ssim']
print(f'Ensemble avg: {total/4:.4f}')
print(f'Platform score: {total/4*100:.2f}')
