#!/usr/bin/env python3
"""Report LLVM code-region unions for explicitly selected function names/files."""
import argparse
import json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('report')
parser.add_argument('functions', nargs='+')
parser.add_argument('--file-suffix', action='append', required=True)
parser.add_argument('--start-line', type=int)
parser.add_argument('--end-line', type=int)
args = parser.parse_args()
with open(args.report) as stream:
    report = json.load(stream)
for requested in args.functions:
    counts = {}
    for unit in report['data']:
        for function in unit['functions']:
            if requested not in function['name']:
                continue
            for region in function['regions']:
                filename = function['filenames'][region[5]]
                if region[7] != 0 or not any(filename.endswith(suffix) for suffix in args.file_suffix):
                    continue
                if args.start_line is not None and region[0] < args.start_line:
                    continue
                if args.end_line is not None and region[2] > args.end_line:
                    continue
                key = (filename, *region[:4])
                counts[key] = max(counts.get(key, 0), region[4])
    if not counts:
        raise SystemExit('no mapped code regions for ' + requested)
    missing = [dict(file=key[0], start_line=key[1], start_column=key[2],
                    end_line=key[3], end_column=key[4])
               for key, count in sorted(counts.items()) if count == 0]
    print(json.dumps(dict(function=requested, covered=len(counts)-len(missing),
                         total=len(counts), uncovered=missing)))
