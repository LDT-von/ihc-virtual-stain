# -*- coding: utf-8
"""Compare all v6 snapshots against v5 best on holdout."""
import sys
ROOT = r'E:\aic\final-ihc'

ckpts = [
    ('fullplus_cd68_v5/final.pt',              'v5_e200'),
    ('fullplus_cd68_v6/snapshot_epoch_050.pt', 'v6_e050'),
    ('fullplus_cd68_v6/snapshot_epoch_100.pt', 'v6_e100'),
    ('fullplus_cd68_v6/snapshot_epoch_150.pt', 'v6_e150'),
    ('fullplus_cd68_v6/snapshot_epoch_200.pt', 'v6_e200'),
    ('fullplus_cd68_v6/snapshot_epoch_250.pt', 'v6_e250'),
    ('fullplus_cd68_v6/final.pt',              'v6_e300'),
]
args = []
for ck, lbl in ckpts:
    args.extend([fr'E:\aic\final-ihc\checkpoints\{ck}', lbl])
import subprocess
cmd = [sys.executable, '-u', fr'{ROOT}\eval_compare.py'] + args
print('compare v5 vs v6 snapshots')
ret = subprocess.run(cmd, cwd=ROOT)
sys.exit(ret.returncode)
