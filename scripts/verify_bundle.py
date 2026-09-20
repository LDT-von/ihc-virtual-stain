"""Verify all delivered source files against BUNDLE_MANIFEST.json."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root/'BUNDLE_MANIFEST.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        path = (root/name).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f'Missing or invalid bundle file: {name}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Changed bundle file: {name}')
    print(json.dumps({'verified_files': len(manifest['files']), 'status': manifest['performance_status']}))


if __name__ == '__main__':
    main()
