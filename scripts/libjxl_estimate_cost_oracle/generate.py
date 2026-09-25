#!/usr/bin/env python3
"""Generate integer-cost goldens by calling the pinned C++ implementation."""
import subprocess
import sys


def samples(w, h, mode):
    state = 0x12345678
    out = []
    for y in range(h):
        for x in range(w):
            state = (state * 1103515245 + 12345) & 0xffffffff
            r = state >> 16
            if mode == 0:
                value = r & 255
            elif mode == 1:
                value = (r & 1) * 255
            elif mode == 2:
                value = ((x + y) % 16) * 17
            elif mode == 3:
                value = 255 if x == w - 1 and y == h - 1 else (x + y) % 15
            elif mode == 4:
                value = (r & 255) * 257 - 32768
            else:
                value = state if state < 0x80000000 else state - 0x100000000
            out.append(value)
    return out


def main():
    requests = []
    for w, h in [(1, 1), (3, 5), (15, 17), (16, 16), (17, 19), (64, 32), (129, 73), (256, 256)]:
        for mode in range(6):
            data = samples(w, h, mode)
            for compact in [0, 1]:
                palette = sorted(set(data))
                if compact:
                    lookup = {v: i for i, v in enumerate(palette)}
                    channels = [(len(palette), 1, palette), (w, h, [lookup[v] for v in data])]
                else:
                    channels = [(w, h, data)]
                # A second channel exercises whole-image fractional accumulation.
                channels.append((3, 1, [17, 0, 65535]))
                name = f'{w}:{h}:{mode}:{compact}'
                row = [name, str(len(channels))]
                for cw, ch, values in channels:
                    row.extend([str(cw), str(ch), *map(str, values)])
                requests.append(' '.join(row))
    result = subprocess.run([sys.argv[1]], input='\n'.join(requests)+'\n', text=True,
                            capture_output=True, check=True)
    sys.stdout.write('# libjxl v0.12.0 a7a9c78; width:height:mode:compact\tcost\n')
    sys.stdout.write(result.stdout)


if __name__ == '__main__':
    main()
