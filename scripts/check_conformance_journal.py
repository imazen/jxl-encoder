#!/usr/bin/env python3
"""Check a completed sweep's journal and optionally a paired control table."""
import argparse
import csv
from pathlib import Path


def rows(path):
    with path.open() as stream:
        data = list(csv.DictReader((line for line in stream if not line.startswith('#')), delimiter='\t'))
    assert data, f'empty results: {path}'
    indices = [int(row['index']) for row in data]
    assert len(indices) == len(set(indices)), f'duplicate cell: {path}'
    assert all(None not in row and all(value is not None for value in row.values()) for row in data), path
    return sorted(data, key=lambda row: int(row['index']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--control', type=Path)
    args = parser.parse_args()
    final = rows(args.out / 'results.tsv')
    journal = rows(args.out / 'results.partial.tsv')
    assert final == journal, 'journal differs from final table'
    if args.control:
        control = rows(args.control)
        def stable(data):
            return [{key: value for key, value in row.items() if key != 'encode_ms'} for row in data]
        assert stable(final) == stable(control), 'cell plan, encoded length or verdict differs from control'
    print(f'{len(final)} journal rows match the final table' + (' and control' if args.control else ''))


if __name__ == '__main__':
    main()
