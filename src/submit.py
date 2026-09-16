"""按赛题阶段打包提交文件。

初赛压缩包的顶层目录为：
results/test/<marker>/<输入名>_fake.jpg

复赛与半决赛会在结果之外加入源码、配置、模型、运行说明和技术报告。
"""
import argparse
import zipfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="打包 AIC 虚拟染色赛题提交文件")
    parser.add_argument("--results-dir", default="results", help="包含 test/<marker>/ 的结果根目录")
    parser.add_argument("--marker", required=True, help="本次提交的目标标记")
    parser.add_argument("--stage", choices=["preliminary", "rematch", "semifinal"], default="preliminary")
    parser.add_argument("--ckpt", help="复赛或半决赛必填：训练得到的模型权重")
    parser.add_argument("--report", help="复赛或半决赛必填：不少于 2000 字的 PDF 技术报告")
    parser.add_argument("--readme", default="README.md", help="运行环境与推理说明文件")
    parser.add_argument("--out-zip", default="submission.zip", help="生成的 ZIP 文件路径")
    return parser.parse_args()


def add_tree(archive: zipfile.ZipFile, path: Path) -> int:
    """保留项目内相对路径写入附加复现材料。"""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        archive.write(path, path.as_posix())
        return 1

    count = 0
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file():
            archive.write(file_path, file_path.as_posix())
            count += 1
    return count


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    target_dir = results_dir / "test" / args.marker
    images = sorted(target_dir.glob("*_fake.jpg"))
    if not images:
        raise FileNotFoundError(
            f"未找到提交图像：{target_dir}。请先运行 src.inference 生成 *_fake.jpg。"
        )

    extras: list[Path] = []
    if args.stage != "preliminary":
        if not args.ckpt or not args.report:
            raise ValueError("复赛和半决赛必须提供 --ckpt 与 --report")
        extras = [Path("src"), Path("configs"), Path(args.ckpt), Path(args.report), Path(args.readme)]

    zip_path = Path(args.out_zip)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for image in images:
            archive.write(image, f"results/test/{args.marker}/{image.name}")
        written = len(images)
        for path in extras:
            written += add_tree(archive, path)

    print(f"[Submit] 阶段：{args.stage}")
    print(f"[Submit] 结果：{len(images)} 张，路径 {target_dir}")
    print(f"[Submit] 已写入 {written} 个文件 -> {zip_path}")


if __name__ == "__main__":
    main()
