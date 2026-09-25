"""Launch 3 bold training experiments with proper resume support."""
import os, subprocess, time

# Find data root
DATA_ROOT = None
for name in os.listdir(r'E:\aic'):
    if any(ord(c) > 127 for c in name):
        full = os.path.join(r'E:\aic', name)
        td = os.path.join(full, 'train', 'DAPI')
        if os.path.exists(td) and len(os.listdir(td)) == 2380:
            DATA_ROOT = full; break

MANIFEST = r'E:\aic\final-ihc\configs\roi_split_semifinal_2026_expanded.json'
PY = r'D:\Anaconda3\python.exe'
SCRIPT = r'E:\aic\final-ihc\train_bold.py'
LOGDIR = r'E:\aic\final-ihc\logs'
CKPTDIR = r'E:\aic\final-ihc\checkpoints'

# A: Continue v3 (lr decays from 1e-5 per cosine over 50 ep)
# B: Vimentin=1.5x boost, init from v3
# C: Balanced weights, init from v3

exps = [
    dict(name='A_v3_continue', output=f'{CKPTDIR}\\fullplus_cd68_v3_continue',
         init=f'{CKPTDIR}\\fullplus_cd68_v3\\final.pt',
         epochs=50, lr='1e-5',
         cd68='3.0', vim='1.0', hla='1.0', cd45ro='1.0',
         desc='Continue v3: lr=1e-5, 50 ep'),
    dict(name='B_vim_boost', output=f'{CKPTDIR}\\w96_vim_boost',
         init=f'{CKPTDIR}\\fullplus_cd68_v3\\final.pt',
         epochs=60, lr='3e-5',
         cd68='3.0', vim='1.5', hla='1.0', cd45ro='1.0',
         desc='Boost Vimentin=1.5x, lr=3e-5, 60 ep'),
    dict(name='C_balanced', output=f'{CKPTDIR}\\w96_balanced',
         init=f'{CKPTDIR}\\fullplus_cd68_v3\\final.pt',
         epochs=60, lr='2e-5',
         cd68='2.0', vim='2.0', hla='1.5', cd45ro='1.0',
         desc='Balanced: CD68=2x, Vim=2x, HLA=1.5x, lr=2e-5, 60 ep'),
]

print(f"DATA_ROOT: {DATA_ROOT}")
for i, exp in enumerate(exps):
    log = f'{LOGDIR}\\{exp["name"]}.log'
    cmd = [PY, SCRIPT,
           '--data-root', DATA_ROOT,
           '--manifest', MANIFEST,
           '--output', exp['output'],
           '--init-checkpoint', exp['init'],
           '--epochs', str(exp['epochs']),
           '--lr', exp['lr'],
           '--cd68-weight', exp['cd68'],
           '--vimentin-weight', exp['vim'],
           '--hla-weight', exp['hla'],
           '--cd45ro-weight', exp['cd45ro'],
           ]
    args_str = ','.join(f'"{c}"' for c in cmd)
    ps = (f'Start-Process -FilePath "{PY}" '
          f'-ArgumentList {args_str} '
          f'-WorkingDirectory "E:\\aic\\final-ihc" '
          f'-RedirectStandardOutput "{log}" '
          f'-RedirectStandardError "{log}.err" '
          f'-NoNewWindow -PassThru | Select-Object Id')
    r = subprocess.run(['powershell', '-Command', ps], capture_output=True, text=True, encoding='utf-8', errors='replace')
    pid = r.stdout.strip() if r.stdout.strip() else '?'
    print(f'{i+1}. {exp["name"]}: PID={pid}')
    print(f'   {exp["desc"]}')
    time.sleep(2)

print(f'\nAll {len(exps)} experiments launched!')
print('Monitor with: Get-Content logs\\<name>.log -Tail 3 -Wait')
