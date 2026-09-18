"""检查训练数据 DAPI vs HLA-DR 均值差异"""
import numpy as np
from PIL import Image
from pathlib import Path

# 抽 100 张
files_d = sorted(Path(r'E:\aic\ihc-data\train\DAPI').glob('*.jpg'))[:100]
files_h = sorted(Path(r'E:\aic\ihc-data\train\HLA-DR').glob('*.jpg'))[:100]

means_d = []
means_h = []

for fd, fh in zip(files_d, files_h):
    img_d = np.array(Image.open(fd).convert('L'))
    img_h = np.array(Image.open(fh).convert('L'))
    means_d.append(img_d.mean() / 255.0 * 2 - 1)  # 转 [-1,1]
    means_h.append(img_h.mean() / 255.0 * 2 - 1)

print(f"DAPI mean (normalized [-1,1]): {np.mean(means_d):+.4f} ± {np.std(means_d):.4f}")
print(f"HLA-DR mean (normalized [-1,1]): {np.mean(means_h):+.4f} ± {np.std(means_h):.4f}")
print(f"v_target = x_0 - ε, x_0 mean: {np.mean(means_h):+.4f}, ε mean: 0")
print(f"v_target mean (theoretical): {np.mean(means_h):+.4f}")
print()
print("Conclusion:")
print(f"  模型需要预测 v ≈ {np.mean(means_h):+.4f} (HLA-DR 的归一化均值)")
print(f"  实际预测 v ≈ -0.65 (更接近 DAPI 输入的负均值 {np.mean(means_d):+.4f})")
print(f"  说明模型把 target 当成了 DAPI 输入本身！")
