"""NAFNet baseline 推理脚本"""
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

from src.data.dataset import DAPItoIHCDataset
from src.metrics.ssim_psnr import to_uint8


class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class NAFNet(nn.Module):
    """匹配 checkpoint 结构的 NAFNet"""
    def __init__(self, in_ch=3, out_ch=3, width=64):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, width, 3, 1, 1), nn.GELU())
        
        self.enc2_conv = nn.Conv2d(width, width*2, 3, 2, 1)
        self.enc2_act = nn.GELU()
        self.enc2_block0_norm = LayerNorm2d(width*2)
        self.enc2_block0_ffn0 = nn.Linear(width*2, width*4)
        self.enc2_block0_ffn2 = nn.Linear(width*2, width*2)
        self.enc2_block0_scale = nn.Parameter(torch.ones(1, width*2, 1, 1))
        self.enc2_block1_norm = LayerNorm2d(width*2)
        self.enc2_block1_ffn0 = nn.Linear(width*2, width*4)
        self.enc2_block1_ffn2 = nn.Linear(width*2, width*2)
        self.enc2_block1_scale = nn.Parameter(torch.ones(1, width*2, 1, 1))
        
        self.enc3_conv = nn.Conv2d(width*2, width*4, 3, 2, 1)
        self.enc3_act = nn.GELU()
        self.enc3_block0_norm = LayerNorm2d(width*4)
        self.enc3_block0_ffn0 = nn.Linear(width*4, width*8)
        self.enc3_block0_ffn2 = nn.Linear(width*4, width*4)
        self.enc3_block0_scale = nn.Parameter(torch.ones(1, width*4, 1, 1))
        self.enc3_block1_norm = LayerNorm2d(width*4)
        self.enc3_block1_ffn0 = nn.Linear(width*4, width*8)
        self.enc3_block1_ffn2 = nn.Linear(width*4, width*4)
        self.enc3_block1_scale = nn.Parameter(torch.ones(1, width*4, 1, 1))
        
        self.enc4_conv = nn.Conv2d(width*4, width*8, 3, 2, 1)
        self.enc4_act = nn.GELU()
        for i in range(3):
            setattr(self, f'enc4_block{i}_norm', LayerNorm2d(width*8))
            setattr(self, f'enc4_block{i}_ffn0', nn.Linear(width*8, width*16))
            setattr(self, f'enc4_block{i}_ffn2', nn.Linear(width*8, width*8))
            setattr(self, f'enc4_block{i}_scale', nn.Parameter(torch.ones(1, width*8, 1, 1)))
        
        self.dec3_upconv = nn.ConvTranspose2d(width*8, width*4, 2, 2)
        self.dec3_act = nn.GELU()
        for i in range(2):
            setattr(self, f'dec3_block{i}_norm', LayerNorm2d(width*4))
            setattr(self, f'dec3_block{i}_ffn0', nn.Linear(width*4, width*8))
            setattr(self, f'dec3_block{i}_ffn2', nn.Linear(width*4, width*4))
            setattr(self, f'dec3_block{i}_scale', nn.Parameter(torch.ones(1, width*4, 1, 1)))
        
        self.dec2_upconv = nn.ConvTranspose2d(width*8, width*2, 2, 2)
        self.dec2_act = nn.GELU()
        for i in range(2):
            setattr(self, f'dec2_block{i}_norm', LayerNorm2d(width*2))
            setattr(self, f'dec2_block{i}_ffn0', nn.Linear(width*2, width*4))
            setattr(self, f'dec2_block{i}_ffn2', nn.Linear(width*2, width*2))
            setattr(self, f'dec2_block{i}_scale', nn.Parameter(torch.ones(1, width*2, 1, 1)))
        
        self.dec1_upconv = nn.ConvTranspose2d(width*4, width, 2, 2)
        self.dec1_act = nn.GELU()
        for i in range(2):
            setattr(self, f'dec1_block{i}_norm', LayerNorm2d(width))
            setattr(self, f'dec1_block{i}_ffn0', nn.Linear(width, width*2))
            setattr(self, f'dec1_block{i}_ffn2', nn.Linear(width, width))
            setattr(self, f'dec1_block{i}_scale', nn.Parameter(torch.ones(1, width, 1, 1)))
        
        self.final0 = nn.Conv2d(width, width, 3, 1, 1)
        self.final_act = nn.GELU()
        self.final2 = nn.Conv2d(width, out_ch, 3, 1, 1)
        self.final_tanh = nn.Tanh()

    def _naf_block(self, x, norm, ffn0, ffn2, scale):
        x_norm = norm(x)
        ffn_out = ffn0(x_norm.permute(0, 2, 3, 1))
        ffn_out = F.gelu(ffn_out)
        ffn_out = ffn2(ffn_out).permute(0, 3, 1, 2)
        return x + ffn_out * scale

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2_act(self.enc2_conv(e1))
        e2 = self._naf_block(e2, self.enc2_block0_norm, self.enc2_block0_ffn0, self.enc2_block0_ffn2, self.enc2_block0_scale)
        e2 = self._naf_block(e2, self.enc2_block1_norm, self.enc2_block1_ffn0, self.enc2_block1_ffn2, self.enc2_block1_scale)
        e3 = self.enc3_act(self.enc3_conv(e2))
        e3 = self._naf_block(e3, self.enc3_block0_norm, self.enc3_block0_ffn0, self.enc3_block0_ffn2, self.enc3_block0_scale)
        e3 = self._naf_block(e3, self.enc3_block1_norm, self.enc3_block1_ffn0, self.enc3_block1_ffn2, self.enc3_block1_scale)
        e4 = self.enc4_act(self.enc4_conv(e3))
        for i in range(3):
            e4 = self._naf_block(e4, getattr(self, f'enc4_block{i}_norm'), getattr(self, f'enc4_block{i}_ffn0'), getattr(self, f'enc4_block{i}_ffn2'), getattr(self, f'enc4_block{i}_scale'))
        d3 = self.dec3_act(self.dec3_upconv(e4))
        for i in range(2):
            d3 = self._naf_block(d3, getattr(self, f'dec3_block{i}_norm'), getattr(self, f'dec3_block{i}_ffn0'), getattr(self, f'dec3_block{i}_ffn2'), getattr(self, f'dec3_block{i}_scale'))
        d3 = torch.cat([d3, e3], 1)
        d2 = self.dec2_act(self.dec2_upconv(d3))
        for i in range(2):
            d2 = self._naf_block(d2, getattr(self, f'dec2_block{i}_norm'), getattr(self, f'dec2_block{i}_ffn0'), getattr(self, f'dec2_block{i}_ffn2'), getattr(self, f'dec2_block{i}_scale'))
        d2 = torch.cat([d2, e2], 1)
        d1 = self.dec1_act(self.dec1_upconv(d2))
        for i in range(2):
            d1 = self._naf_block(d1, getattr(self, f'dec1_block{i}_norm'), getattr(self, f'dec1_block{i}_ffn0'), getattr(self, f'dec1_block{i}_ffn2'), getattr(self, f'dec1_block{i}_scale'))
        d1 = torch.cat([d1, e1], 1)
        out = self.final_act(self.final0(d1))
        out = self.final_tanh(self.final2(out))
        return out


def load_checkpoint(model, ck):
    state = ck['model']
    model_state = {}
    model_state['enc1.0.weight'] = state['enc1.0.weight']
    model_state['enc1.0.bias'] = state['enc1.0.bias']
    model_state['enc2_conv.weight'] = state['enc2.0.weight']
    model_state['enc2_conv.bias'] = state['enc2.0.bias']
    model_state['enc2_block0_norm.norm.weight'] = state['enc2.2.norm.norm.weight']
    model_state['enc2_block0_norm.norm.bias'] = state['enc2.2.norm.norm.bias']
    model_state['enc2_block0_ffn0.weight'] = state['enc2.2.ffn.0.weight']
    model_state['enc2_block0_ffn0.bias'] = state['enc2.2.ffn.0.bias']
    model_state['enc2_block0_ffn2.weight'] = state['enc2.2.ffn.2.weight']
    model_state['enc2_block0_ffn2.bias'] = state['enc2.2.ffn.2.bias']
    model_state['enc2_block0_scale'] = state['enc2.2.res_scale']
    model_state['enc2_block1_norm.norm.weight'] = state['enc2.3.norm.norm.weight']
    model_state['enc2_block1_norm.norm.bias'] = state['enc2.3.norm.norm.bias']
    model_state['enc2_block1_ffn0.weight'] = state['enc2.3.ffn.0.weight']
    model_state['enc2_block1_ffn0.bias'] = state['enc2.3.ffn.0.bias']
    model_state['enc2_block1_ffn2.weight'] = state['enc2.3.ffn.2.weight']
    model_state['enc2_block1_ffn2.bias'] = state['enc2.3.ffn.2.bias']
    model_state['enc2_block1_scale'] = state['enc2.3.res_scale']
    model_state['enc3_conv.weight'] = state['enc3.0.weight']
    model_state['enc3_conv.bias'] = state['enc3.0.bias']
    for i, idx in enumerate(['2', '3']):
        base = f'enc3.{idx}'
        b = f'enc3_block{i}'
        model_state[f'{b}_norm.norm.weight'] = state[f'{base}.norm.norm.weight']
        model_state[f'{b}_norm.norm.bias'] = state[f'{base}.norm.norm.bias']
        model_state[f'{b}_ffn0.weight'] = state[f'{base}.ffn.0.weight']
        model_state[f'{b}_ffn0.bias'] = state[f'{base}.ffn.0.bias']
        model_state[f'{b}_ffn2.weight'] = state[f'{base}.ffn.2.weight']
        model_state[f'{b}_ffn2.bias'] = state[f'{base}.ffn.2.bias']
        model_state[f'{b}_scale'] = state[f'{base}.res_scale']
    model_state['enc4_conv.weight'] = state['enc4.0.weight']
    model_state['enc4_conv.bias'] = state['enc4.0.bias']
    for i, idx in enumerate(['2', '3', '4']):
        base = f'enc4.{idx}'
        b = f'enc4_block{i}'
        model_state[f'{b}_norm.norm.weight'] = state[f'{base}.norm.norm.weight']
        model_state[f'{b}_norm.norm.bias'] = state[f'{base}.norm.norm.bias']
        model_state[f'{b}_ffn0.weight'] = state[f'{base}.ffn.0.weight']
        model_state[f'{b}_ffn0.bias'] = state[f'{base}.ffn.0.bias']
        model_state[f'{b}_ffn2.weight'] = state[f'{base}.ffn.2.weight']
        model_state[f'{b}_ffn2.bias'] = state[f'{base}.ffn.2.bias']
        model_state[f'{b}_scale'] = state[f'{base}.res_scale']
    model_state['dec3_upconv.weight'] = state['dec3.0.weight']
    model_state['dec3_upconv.bias'] = state['dec3.0.bias']
    for i, idx in enumerate(['2', '3']):
        base = f'dec3.{idx}'
        b = f'dec3_block{i}'
        model_state[f'{b}_norm.norm.weight'] = state[f'{base}.norm.norm.weight']
        model_state[f'{b}_norm.norm.bias'] = state[f'{base}.norm.norm.bias']
        model_state[f'{b}_ffn0.weight'] = state[f'{base}.ffn.0.weight']
        model_state[f'{b}_ffn0.bias'] = state[f'{base}.ffn.0.bias']
        model_state[f'{b}_ffn2.weight'] = state[f'{base}.ffn.2.weight']
        model_state[f'{b}_ffn2.bias'] = state[f'{base}.ffn.2.bias']
        model_state[f'{b}_scale'] = state[f'{base}.res_scale']
    model_state['dec2_upconv.weight'] = state['dec2.0.weight']
    model_state['dec2_upconv.bias'] = state['dec2.0.bias']
    for i, idx in enumerate(['2', '3']):
        base = f'dec2.{idx}'
        b = f'dec2_block{i}'
        model_state[f'{b}_norm.norm.weight'] = state[f'{base}.norm.norm.weight']
        model_state[f'{b}_norm.norm.bias'] = state[f'{base}.norm.norm.bias']
        model_state[f'{b}_ffn0.weight'] = state[f'{base}.ffn.0.weight']
        model_state[f'{b}_ffn0.bias'] = state[f'{base}.ffn.0.bias']
        model_state[f'{b}_ffn2.weight'] = state[f'{base}.ffn.2.weight']
        model_state[f'{b}_ffn2.bias'] = state[f'{base}.ffn.2.bias']
        model_state[f'{b}_scale'] = state[f'{base}.res_scale']
    model_state['dec1_upconv.weight'] = state['dec1.0.weight']
    model_state['dec1_upconv.bias'] = state['dec1.0.bias']
    for i, idx in enumerate(['2', '3']):
        base = f'dec1.{idx}'
        b = f'dec1_block{i}'
        model_state[f'{b}_norm.norm.weight'] = state[f'{base}.norm.norm.weight']
        model_state[f'{b}_norm.norm.bias'] = state[f'{base}.norm.norm.bias']
        model_state[f'{b}_ffn0.weight'] = state[f'{base}.ffn.0.weight']
        model_state[f'{b}_ffn0.bias'] = state[f'{base}.ffn.0.bias']
        model_state[f'{b}_ffn2.weight'] = state[f'{base}.ffn.2.weight']
        model_state[f'{b}_ffn2.bias'] = state[f'{base}.ffn.2.bias']
        model_state[f'{b}_scale'] = state[f'{base}.res_scale']
    model_state['final0.weight'] = state['final.0.weight']
    model_state['final0.bias'] = state['final.0.bias']
    model_state['final2.weight'] = state['final.2.weight']
    model_state['final2.bias'] = state['final.2.bias']
    model.load_state_dict(model_state, strict=True)
    print(f'Loaded {len(model_state)} weights')


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--marker', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--split', default='test')
    p.add_argument('--output', default='results')
    p.add_argument('--batch-size', type=int, default=8)
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f'[NAFNet Infer] marker={args.marker}')
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model = NAFNet(in_ch=3, out_ch=3, width=64).to(device)
    load_checkpoint(model, ck)
    model.eval()
    print(f'[Model] epoch={ck.get("epoch", "?")} val_ssim={ck.get("val_ssim", 0):.4f}')
    
    ds = DAPItoIHCDataset(args.data_root, args.marker, split=args.split, patch_size=256)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f'[Data] {len(ds)} samples')
    
    out_dir = Path(args.output) / args.split / args.marker
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.inference_mode():
        for batch in loader:
            dapi = batch['dapi'].to(device)
            fake = model(dapi)
            for i in range(fake.shape[0]):
                img = Image.fromarray(to_uint8(fake[i].cpu().numpy()))
                img.save(out_dir / f"{batch['name'][i]}_fake.jpg", quality=95)
    
    count = len(list(out_dir.glob('*_fake.jpg')))
    print(f'[Done] {count} images')


if __name__ == '__main__':
    main()
