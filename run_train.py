"""Quick training launcher for all IHC models.

Usage:
    python run_train.py --model enhanced_unet --data-root DATA_ROOT --manifest MANIFEST
    python run_train.py --model transformer --model-type hybrid --data-root DATA_ROOT --manifest MANIFEST
    
Models available:
- enhanced_unet: Enhanced UNet++ with dense connections
- transformer: Transformer-based model (restormer/simple/hybrid)
- marigold: Latent diffusion model (SOTA, requires more VRAM)

Author: AI Assistant
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.roi_manifest import SEMIFINAL_SEED


def main():
    parser = argparse.ArgumentParser(description='Quick IHC Model Training')
    
    parser.add_argument('--model', required=True, 
                       choices=['enhanced_unet', 'transformer', 'marigold'],
                       help='Model type')
    parser.add_argument('--data-root', required=True, help='Data root directory')
    parser.add_argument('--manifest', required=True, help='ROI manifest JSON')
    parser.add_argument('--output', required=True, help='Output directory')
    
    # Common args
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=SEMIFINAL_SEED)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--tta', type=int, default=4, choices=[1, 4, 8])
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--train-limit', type=int, default=0)
    parser.add_argument('--val-limit', type=int, default=0)
    
    # Model-specific args
    parser.add_argument('--base-channels', type=int, default=64)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--model-type', default='hybrid', 
                       choices=['restormer', 'simple', 'hybrid'],
                       help='Specific transformer architecture')
    
    args = parser.parse_args()
    
    if args.model == 'enhanced_unet':
        from src.train_enhanced_unet import train
        print("Launching Enhanced UNet++ training...")
        train(args)
    
    elif args.model == 'transformer':
        from src.train_transformer_ihc import train
        print(f"Launching Transformer ({args.model_type}) training...")
        train(args)
    
    elif args.model == 'marigold':
        print("Launching Marigold Latent Diffusion training...")
        from src.train_marigold_ihc import train_stage1
        train_stage1(args)
    
    else:
        raise ValueError(f"Unknown model: {args.model}")


if __name__ == '__main__':
    main()
