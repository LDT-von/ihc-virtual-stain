"""打包提交：根据官方要求打包 submission.zip

注意：实际提交目录结构请以最新官方细则为准
"""
import argparse
import zipfile
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--marker", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True, help="推理输出目录（含 images/）")
    p.add_argument("--out_zip", type=str, default="submission.zip")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.exists():
        raise FileNotFoundError(out_dir)

    zip_path = Path(args.out_zip)

    # 包含：images/、源代码、模型权重（按要求）
    files_to_pack = []
    img_dir = out_dir / "images"
    if img_dir.exists():
        files_to_pack += sorted(img_dir.rglob("*"))
    files_to_pack += [Path("src"), Path("configs"), Path(args.ckpt)]

    print(f"[Submit] 打包 {len(files_to_pack)} 项到 {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files_to_pack:
            if p.is_dir():
                continue
            arcname = p.as_posix()
            zf.write(p, arcname)
            print(f"  + {arcname}")
    print(f"[Submit] 完成 -> {zip_path}")


if __name__ == "__main__":
    main()