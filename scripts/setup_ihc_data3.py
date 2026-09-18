"""用 val 当 train 重训（不破坏原始数据）

做法：建 ihc-data3/train/{DAPI,HLA-DR} 把 val 内容复制过去
"""
import shutil
from pathlib import Path

src = Path(r'E:\aic\ihc-data\val')
dst = Path(r'E:\aic\ihc-data3\train')
dst.mkdir(parents=True, exist_ok=True)

# 用 junction 而不是复制（更快）
for sub in ['DAPI', 'HLA-DR']:
    target = dst / sub
    if target.exists():
        shutil.rmtree(target)
    # junction (符号链接)
    src_sub = src / sub
    import subprocess
    subprocess.run(['cmd', '/c', 'mklink', '/J', str(target), str(src_sub)], check=True)
    print(f"Linked: {target} -> {src_sub}")

# 同时把 val 目录指到 ihc-data3/val (junction)
val_dst = Path(r'E:\aic\ihc-data3\val')
if val_dst.exists():
    shutil.rmtree(val_dst)
import subprocess
subprocess.run(['cmd', '/c', 'mklink', '/J', str(val_dst), str(src)], check=True)
print(f"Linked: {val_dst} -> {src}")

# 也加 test
src_test = Path(r'E:\aic\ihc-data\test')
dst_test = Path(r'E:\aic\ihc-data3\test')
if dst_test.exists():
    shutil.rmtree(dst_test)
import subprocess
subprocess.run(['cmd', '/c', 'mklink', '/J', str(dst_test), str(src_test)], check=True)
print(f"Linked: {dst_test} -> {src_test}")
