import torch
from src.models.losses import PyramidLoss, PyramidL1SSIMLoss, GaussianPyramid

gp = GaussianPyramid(levels=4)
x = torch.rand(2, 3, 256, 256) * 2 - 1
y = torch.rand(2, 3, 256, 256) * 2 - 1
p = gp(x)
print('Pyramid shapes:', [t.shape for t in p])

pl = PyramidLoss(levels=4)
print('PyramidL1:', pl(x, y).item())

plssim = PyramidL1SSIMLoss()
print('PyramidL1+SSIM:', plssim(x, y).item())

# backward check
plssim(x, y).backward()
print('backward OK')
