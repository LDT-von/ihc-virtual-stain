"""Inspect actual checkpoint architecture/provenance without exporting tensor values."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def inspect(path):
    path = Path(path)
    ck = torch.load(path,map_location='cpu',weights_only=False)
    run = ck.get('run',{})
    state = ck.get('ema',ck.get('model',ck.get('state_dict',{})))
    keys = set(state)
    if any(key.startswith('baseline.') for key in keys):
        family = 'AnchoredIHC'
    elif 'stem.weight' in keys and any(key.startswith('heads.') for key in keys):
        family = 'MarkerContextNet'
    elif 'stem.weight' in keys and any(key.startswith('decoders.0.head.') for key in keys):
        family = 'MarkerSpecificNet'
    elif any(key.startswith('generator.input_conv.') for key in keys):
        family = 'Pix2PixGAN (generator and possible discriminator)'
    elif any('backbone' in key or 'encoder.layer1' in key for key in keys):
        family = 'ResNet-family candidate; check saved source/config'
    else:
        family = 'unknown; do not infer from directory name'
    return {'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'detected_family':family,'epoch':ck.get('epoch'),'state_key_count':len(keys),
            'model_config':run.get('model_config',ck.get('cfg',{}).get('model')),
            'args':run.get('args'),'split_sha256':run.get('split_sha256'),
            'train_count':len(run.get('train_names',[])),'val_count':len(run.get('val_names',[])),
            'git_head':run.get('environment',{}).get('git_head'),
            'source_sha256':run.get('environment',{}).get('source_sha256'),
            'initial_checkpoint_sha256':run.get('initial_checkpoint_sha256')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('paths',nargs='+',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    files=[]
    for path in args.paths:
        if path.is_dir():
            best=sorted(path.rglob('best.pt'))
            files.extend(best or sorted(path.rglob('final.pt')))
        else:
            files.append(path)
    if not files:
        raise FileNotFoundError('No best.pt/final.pt checkpoints at the supplied paths')
    report=[inspect(path) for path in sorted(set(files))]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps([{'path':row['path'],'family':row['detected_family']} for row in report]))


if __name__=='__main__':
    main()
