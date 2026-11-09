#!/usr/bin/env python3
"""Parallel-download the public CSV mirror of ULB/Worldline credit-card fraud data."""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

URL = 'https://huggingface.co/datasets/David-Egea/Creditcard-fraud-detection/resolve/main/creditcard.csv?download=true'
CHUNK_SIZE = 8 * 1024 * 1024
WORKERS = 8


def download_part(start: int, end: int, part_path: Path) -> int:
    headers = {'Range': f'bytes={start}-{end}', 'User-Agent': 'Mozilla/5.0 FraudDetectionEducation/1.0'}
    with requests.get(URL, headers=headers, stream=True, timeout=(20, 180)) as response:
        if response.status_code != 206:
            raise RuntimeError(f'Expected HTTP 206 for bytes {start}-{end}, got {response.status_code}')
        expected_range = f'bytes {start}-{end}/'
        if not (response.headers.get('Content-Range') or '').startswith(expected_range):
            raise RuntimeError(f'Unexpected Content-Range for chunk {start}-{end}: {response.headers.get("Content-Range")}')
        size = 0
        with part_path.open('wb') as out:
            for block in response.iter_content(1024 * 1024):
                if block:
                    out.write(block)
                    size += len(block)
    expected_size = end - start + 1
    if size != expected_size:
        raise RuntimeError(f'Chunk {start}-{end}: expected {expected_size} bytes, got {size}')
    return size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/raw/creditcard.csv')
    args = parser.parse_args()
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {'User-Agent': 'Mozilla/5.0 FraudDetectionEducation/1.0'}
    response = requests.head(URL, allow_redirects=True, headers=headers, timeout=30)
    response.raise_for_status()
    total_size = int(response.headers.get('Content-Length', '0'))
    accepts_ranges = 'bytes' in (response.headers.get('Accept-Ranges') or '').lower()
    if not total_size or not accepts_ranges:
        raise RuntimeError('Mirror did not provide file size and byte-range support; refusing an unverified partial download.')

    parts_dir = dest.parent / (dest.name + '.parts')
    parts_dir.mkdir(exist_ok=True)
    ranges = [(start, min(start + CHUNK_SIZE - 1, total_size - 1)) for start in range(0, total_size, CHUNK_SIZE)]
    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {
                pool.submit(download_part, start, end, parts_dir / f'{start:012d}.part'): (start, end)
                for start, end in ranges
            }
            for future in as_completed(futures):
                completed += future.result()
                print(f'\rDownloaded and verified {completed / total_size:.0%} ({completed:,}/{total_size:,} bytes)', end='', flush=True)
        temp = dest.with_suffix(dest.suffix + '.tmp')
        with temp.open('wb') as out:
            for start, _end in ranges:
                with (parts_dir / f'{start:012d}.part').open('rb') as part:
                    while True:
                        block = part.read(1024 * 1024)
                        if not block:
                            break
                        out.write(block)
        if temp.stat().st_size != total_size:
            raise RuntimeError('Assembled file size does not match the source.')
        os.replace(temp, dest)
    finally:
        for p in parts_dir.glob('*.part'):
            p.unlink(missing_ok=True)
        parts_dir.rmdir()
    print(f'\nSaved {total_size:,} bytes to {dest.resolve()}')


if __name__ == '__main__':
    main()
