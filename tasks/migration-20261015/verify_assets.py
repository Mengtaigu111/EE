#!/usr/bin/env python3
"""Read-only verification of transported bytes; historical provenance is separate."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/'), help='Filesystem root, or a staging root for inspection')
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('transfer-manifest.json'))
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text())
    failed = []
    for row in data['files']:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe manifest path: ' + row['path'])
        path = args.root / relative
        if not path.is_file() or path.stat().st_size != row['bytes']:
            failed.append(row['path'] + ': missing/size mismatch')
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != row['sha256']:
            failed.append(row['path'] + ': SHA256 mismatch')
    for error in failed:
        print(error)
    print(json.dumps({'checked': len(data['files']), 'failed': len(failed), 'bytes': data['bytes']}))
    raise SystemExit(bool(failed))


if __name__ == '__main__':
    main()
