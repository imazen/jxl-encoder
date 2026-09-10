#!/usr/bin/env python3
"""Process-wall + bytes A/B of cjxl-rs against cjxl v0.12, per effort.

Both arms read the same metadata-stripped PNG and write a real file, so process
start, PNG decode and file write sit in BOTH denominators. That is the fair
comparison for "how long does the user wait"; it is NOT an encode-only ratio and
must not be quoted as one.

The arms are ALTERNATED inside each repeat and the leading arm flips every
repeat, because block-ordered repeats inverted the sign of an 8 % effect on
short cells during T3 (2026-08-31).

Usage:
  scripts/ladder_vs_cjxl.py --ours <cjxl-rs> --cjxl <cjxl> --out <tsv>
      [--reps 5] [--efforts 3,5,7,9] [--distances 1.0,4.0] [--threads 1,8]
      <png>...
"""
import argparse, os, subprocess, sys, tempfile, time

def run(cmd, out):
    t = time.perf_counter()
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ms = (time.perf_counter() - t) * 1000.0
    if r.returncode != 0:
        return None, None
    return ms, os.path.getsize(out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ours', required=True)
    p.add_argument('--cjxl', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--reps', type=int, default=5)
    p.add_argument('--efforts', default='3,5,7,9')
    p.add_argument('--distances', default='1.0,4.0')
    p.add_argument('--threads', default='1,8')
    p.add_argument('--cjxl-flags', default='libjxl',
                   choices=['libjxl', 'ours'],
                   help="flag dialect for the --cjxl arm; use 'ours' to A/B two "
                        "cjxl-rs builds against each other")
    p.add_argument('images', nargs='+')
    a = p.parse_args()
    # A wrong --ours/--cjxl path used to produce a header-only TSV, one FAILED
    # line per cell on stderr, and exit code 0 — an hour of measurement that
    # looked like it ran. Prove both binaries encode BEFORE the grid starts.
    # (2026-09-10: `--cjxl .ci-libjxl/tools/cjxl` from the wrong working
    # directory failed all 224 cells exactly this way.)
    for label, exe in (('--ours', a.ours), ('--cjxl', a.cjxl)):
        if not (os.path.isfile(exe) and os.access(exe, os.X_OK)):
            sys.exit(f'{label}: not an executable file: {exe}')

    efforts = [int(x) for x in a.efforts.split(',')]
    dists = [float(x) for x in a.distances.split(',')]
    threads = [int(x) for x in a.threads.split(',')]

    tmp = tempfile.mkdtemp(prefix='ladder', dir=os.path.expanduser('~/tmp'))
    o_out, c_out = os.path.join(tmp, 'o.jxl'), os.path.join(tmp, 'c.jxl')

    probe = a.images[0]
    for label, exe, thr in (('--ours', a.ours, ['--threads', '1']),
                            ('--cjxl', a.cjxl,
                             ['--threads', '1'] if a.cjxl_flags == 'ours'
                             else ['--num_threads=1'])):
        ms, _ = run([exe, probe, o_out, '-e', '3', '-d', '1.0'] + thr, o_out)
        if ms is None:
            sys.exit(f'{label}: {exe} failed to encode {probe} — check the path, '
                     f'the flag dialect (--cjxl-flags) and the input.')

    failed = 0
    with open(a.out, 'w') as f:
        f.write('image\teffort\tdistance\tthreads\tours_ms\tcjxl_ms\twall_ratio\t'
                'ours_bytes\tcjxl_bytes\tbyte_ratio\n')
        for img in a.images:
            tag = os.path.basename(img).rsplit('.', 1)[0]
            for e in efforts:
                for d in dists:
                    for t in threads:
                        mo = mc = float('inf'); bo = bc = None
                        for rep in range(a.reps):
                            order = ['o', 'c'] if rep % 2 == 0 else ['c', 'o']
                            for arm in order:
                                if arm == 'o':
                                    cmd = ['nice', '-n', '19', a.ours, img, o_out,
                                           '-e', str(e), '-d', str(d),
                                           '--threads', str(t)]
                                    ms, b = run(cmd, o_out)
                                    if ms is None:
                                        continue
                                    mo = min(mo, ms); bo = b
                                else:
                                    thr = (['--threads', str(t)]
                                           if a.cjxl_flags == 'ours'
                                           else [f'--num_threads={t}'])
                                    cmd = ['nice', '-n', '19', a.cjxl, img, c_out,
                                           '-e', str(e), '-d', str(d)] + thr
                                    ms, b = run(cmd, c_out)
                                    if ms is None:
                                        continue
                                    mc = min(mc, ms); bc = b
                        if bo is None or bc is None:
                            failed += 1
                            print(f'FAILED {tag} e{e} d{d} t{t}', file=sys.stderr)
                            continue
                        f.write(f'{tag}\t{e}\t{d}\t{t}\t{mo:.0f}\t{mc:.0f}\t'
                                f'{mo/mc:.3f}\t{bo}\t{bc}\t{bo/bc:.4f}\n')
                        f.flush()
                        print(f'{tag} e{e} d{d} t{t}: {mo:.0f} vs {mc:.0f} ms '
                              f'({mo/mc:.2f}x), {bo} vs {bc} B ({bo/bc:.3f}x)',
                              file=sys.stderr)
    if failed:
        print(f'wrote {a.out} with {failed} FAILED cells', file=sys.stderr)
        sys.exit(1)
    print(f'wrote {a.out}', file=sys.stderr)

main()
