"""Loss functions for single-marker virtual staining.

Key insight: SSIM is the dominant metric (weight 0.7 in the scoring formula).
We need losses that directly optimize structural similarity, not just pixel-wise MSE.
"""
import torch
import torch.nn.functional as F


def local_ssim_4d(pred, target, window=7, C1=0.01**2, C2=0.03**2):
    """
    Compute SSIM over 4D (N,1,H,W) tensors.
    Uses uniform window, sample covariance, matches skimage.measure.compare_ssim.
    Returns per-image score (mean over spatial).
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")
    if pred.ndim != 4 or pred.shape[1] != 1:
        raise ValueError(f"Expected (N,1,H,W), got {pred.shape}")
    if min(pred.shape[-2:]) < window:
        raise ValueError(f"Image too small for window={window}")
    
    with torch.autocast(device_type=pred.device.type, enabled=False):
        p, t = pred.float(), target.float()
        # Compute local statistics with uniform average pooling
        pad = window // 2
        cat = torch.cat([p, t, p*p, t*t, p*t], dim=1)
        mp, mt, pp, tt, pt = F.avg_pool2d(cat, window, stride=1, padding=pad).chunk(5, dim=1)
        # Sample variance (Bessel correction)
        vp = (pp - mp*mp) * window*window / (window*window - 1)
        vt = (tt - mt*mt) * window*window / (window*window - 1)
        cov = (pt - mp*mt) * window*window / (window*window - 1)
        
        # SSIM formula
        num = (2*mp*mt + C1) * (2*cov + C2)
        den = (mp*mp + mt*mt + C1) * (vp + vt + C2)
        ssim_map = num / den.clamp_min(1e-12)
        
        # Crop border (same as skimage)
        border = window // 2
        ssim_map = ssim_map[..., border:-border, border:-border]
        return ssim_map.mean(dim=(-2, -1))


def ssim_loss(pred, target):
    """1 - SSIM, averaged over batch and channels."""
    return 1.0 - local_ssim_4d(pred, target).mean()


def reconstruction_loss(pred, target, ssim_weight=1.0, l1_weight=0.5, mse_weight=1.0, multiscale_weight=0.1):
    """
    SSIM + multi-scale L1 loss for single marker.
    
    Args:
        pred: (N,1,H,W) in [0,1]
        target: (N,1,H,W) in [0,1]
        ssim_weight: weight for SSIM term
        l1_weight: weight for L1 term
        mse_weight: weight for MSE term  
        multiscale_weight: weight for multi-scale L1 (2x, 4x downsampled)
    """
    ssim = ssim_loss(pred, target)
    
    diff = (pred - target).abs()
    l1 = diff.mean()
    mse = (pred - target).square().mean()
    
    loss = ssim_weight * ssim + l1_weight * l1 + mse_weight * mse
    
    # Multi-scale supervision
    for scale in (2, 4):
        down_pred = F.avg_pool2d(pred, scale)
        down_target = F.avg_pool2d(target, scale)
        loss = loss + multiscale_weight * (down_pred - down_target).abs().mean()
    
    return loss


def mixed_perceptual_loss(pred, target, edge_weight=0.5):
    """
    SSIM + gradient (edge) loss.
    Gradient loss helps with sharp edges and fine structures.
    """
    ssim = ssim_loss(pred, target)
    
    # Sobel gradients
    def gradient(x):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        gx = F.conv2d(x, sobel_x, padding=1)
        gy = F.conv2d(x, sobel_y, padding=1)
        return gx, gy
    
    px, py = gradient(pred)
    tx, ty = gradient(target)
    edge = (px - tx).abs().mean() + (py - ty).abs().mean()
    
    return ssim + edge_weight * edge


def ms_ssim_loss(pred, target, window=3):
    """
    Multi-scale SSIM: compute SSIM at multiple resolutions and combine.
    More robust to scale variations.
    """
    total = 0.0
    weight = 1.0
    current_p, current_t = pred, target
    
    for _ in range(4):  # 4 scales
        ssim_val = local_ssim_4d(current_p, current_t, window=window)
        total += weight * (1.0 - ssim_val)
        
        if current_p.shape[-1] < 32 or current_p.shape[-2] < 32:
            break
        
        # Downsample by 2
        current_p = F.avg_pool2d(current_p, 2)
        current_t = F.avg_pool2d(current_t, 2)
        weight *= 0.5
    
    return total
