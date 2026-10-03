"""Quick training launcher for all new IHC models.

Usage:
    python run_quick_train.py --model enhanced_unet --data-root DATA_ROOT --manifest MANIFEST
    
Models:
    - enhanced_unet: Enhanced UNet++ with deep supervision
    - hybrid: Hybrid CNN-Attention model
    - attention_unet: Attention UNet with SE blocks

Author: AI Assistant
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.roi_manifest import SEMIFINAL_SEED


def main():
    parser = argparse.ArgumentParser(description='Quick IHC Model Training')
    
    parser.add_argument('--model', required=True, 
                       choices=['enhanced_unet', 'hybrid', 'attention_unet'],
                       help='Model type')
    parser.add_argument('--data-root', required=True, help='Data root directory')
    parser.add_argument('--manifest', required=True, help='ROI manifest JSON')
    parser.add_argument('--output', default='checkpoints/quick_run', help='Output directory')
    
    # Training args
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--base-channels', type=int, default=64)
    parser.add_argument('--tta', type=int, default=4, choices=[1, 4, 8])
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--train-limit', type=int, default=0)
    parser.add_argument('--val-limit', type=int, default=0)
    parser.add_argument('--seed', type=int, default=SEMIFINAL_SEED)
    parser.add_argument('--warmup-epochs', type=int, default=3)
    
    args = parser.parse_args()
    
    if args.model == 'enhanced_unet':
        from src.train_enhanced_unet import train
    elif args.model in ['hybrid', 'attention_unet']:
        from src.train_transformer_ihc import train
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    print(f"=" * 60)
    print(f"Training: {args.model}")
    print(f"Data: {args.data_root}")
    print(f"Output: {args.output}")
    print(f"=" * 60)
    
    train(args)


if __name__ == '__main__':
    main()
