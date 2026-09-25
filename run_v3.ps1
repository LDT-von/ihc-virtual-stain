# Run fullplus_cd68 50 epochs training
cd E:\aic\final-ihc

$env:PYTHONPATH = "E:\aic\final-ihc"

D:\Anaconda3\python.exe `
    train_fullplus_cd68.py `
    --data-root "E:\aic\复赛数据集(包括训练集和测试集输入)" `
    --manifest configs/roi_split_semifinal_2026_expanded.json `
    --init-checkpoint checkpoints/ultimate_w96_expanded/baseline/best.pt `
    --output checkpoints/fullplus_cd68_v3 `
    --epochs 50 `
    --batch-size 8 `
    --lr 3e-5 `
    --cd68-weight 3.0
