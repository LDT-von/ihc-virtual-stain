"""Verify a complete final inference folder and package only official images."""
import argparse
import json
import sys
import zipfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.roi_manifest import MARKERS


def package(input_dir, prediction_dir, destination, markers=MARKERS):
    input_dir, prediction_dir, destination = map(Path, (input_dir, prediction_dir, destination))
    inputs = sorted(input_dir.glob('*.jpg'))
    if not inputs:
        raise ValueError('No test inputs')
    if destination.exists():
        raise FileExistsError(destination)
    expected = {}
    for path in inputs:
        with Image.open(path) as im:
            if im.size != (256, 256):
                raise ValueError(f'Official test input must be 256x256: {path}')
            expected[path.stem+'_fake.jpg'] = im.size
    if len(expected) != len(inputs):
        raise ValueError('Duplicate input names')
    if not markers or len(set(markers)) != len(markers) or set(markers)-set(MARKERS):
        raise ValueError('Invalid marker selection')
    files = []
    for marker in markers:
        folder = prediction_dir/'results'/'test'/marker
        actual = {p.name for p in folder.iterdir() if p.is_file()} if folder.exists() else set()
        if actual != set(expected):
            raise ValueError(f'{marker}: missing={len(set(expected)-actual)}, extra={len(actual-set(expected))}')
        for name in sorted(expected):
            path = folder/name
            with Image.open(path) as im:
                if im.format != 'JPEG' or im.size != expected[name] or im.mode != 'L':
                    raise ValueError(f'Invalid prediction image: {path}')
                im.load()
            files.append((path, f'results/test/{marker}/{name}'))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix+'.tmp')
    if temporary.exists():
        raise FileExistsError(temporary)
    with zipfile.ZipFile(temporary, 'x', zipfile.ZIP_DEFLATED) as archive:
        for path, member in files:
            archive.write(path, member)
    temporary.replace(destination)
    return {'images_per_marker': len(inputs), 'markers': list(markers), 'total_images': len(files),
            'zip': str(destination.resolve()), 'status': 'packaged locally; not submitted'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='test/DAPI directory')
    p.add_argument('--prediction-dir', required=True, help='Final inference output containing results/')
    p.add_argument('--output', required=True)
    p.add_argument('--markers', nargs='+', choices=MARKERS, default=list(MARKERS))
    args = p.parse_args()
    print(json.dumps(package(args.input, args.prediction_dir, args.output, args.markers)))


if __name__ == '__main__':
    main()
