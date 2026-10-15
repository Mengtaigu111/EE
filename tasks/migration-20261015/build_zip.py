#!/usr/bin/env python3
"""Package the project-only inventory and migration tools; never package weights."""
import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    project = here.parents[1]
    manifest = json.loads((here / 'transfer-manifest.json').read_text())
    files = {}
    for row in manifest['files']:
        absolute = Path('/') / row['path']
        if not absolute.is_relative_to(project):
            continue
        files[absolute] = row['sha256']
    for path in here.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            files[path] = None
    if args.output.resolve().is_relative_to(here):
        raise ValueError('Archive must be outside the migration tools directory')
    started = time.monotonic()
    with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as archive:
        for path, expected in sorted(files.items()):
            if expected is not None:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != expected:
                    raise ValueError('Source changed since inventory: ' + str(path))
            archive.write(path, 'EXCEEDS/' + str(path.relative_to(project)))
    with zipfile.ZipFile(args.output) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError('Bad archive member: ' + bad)
    digest = hashlib.sha256()
    with args.output.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    args.output.with_suffix(args.output.suffix + '.sha256').write_text(digest.hexdigest() + '  ' + args.output.name + '\n')
    print(json.dumps({'files': len(files), 'bytes': args.output.stat().st_size,
                      'seconds': round(time.monotonic() - started, 2), 'sha256': digest.hexdigest()}), flush=True)


if __name__ == '__main__':
    main()
