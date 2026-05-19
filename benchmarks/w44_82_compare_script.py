import csv

# W44-79 ledger baseline (before W44-82)
ledger_path = "/home/lilith/work/zen/jxl-encoder--w44-82-custom-orders-gate/benchmarks/cjxl_parity_ledger_2026-05-19_w44_79.tsv"

baseline = {}
with open(ledger_path) as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        if row['effort'] != '7':
            continue
        d = float(row['distance'])
        if d not in (3.0, 4.0, 5.0, 6.0):
            continue
        img = row['image'].replace('.png', '')
        baseline[(img, d)] = (int(row['jxl_bytes']), int(row['cjxl_bytes']))

# Read new bytes
print(f"{'image':<10} {'d':<5} {'old':>8} {'new':>8} {'delta':>8} {'delta_pct':>10} {'cjxl':>8} {'new_vs_cjxl':>11}")
total_delta = 0
total_open_delta = 0
open_cells = {('1420710',5.0),('1531677',5.0),('1531677',6.0),('1420710',6.0)}
n_open = 0
n_open_close_candidates = 0  # bytes <= cjxl
flips_to_fixed = 0  # bytes_delta% goes <= 1.5 (FIXED threshold)
new_open = 0
with open("/tmp/w44_82_bytes_new.tsv") as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        img = row['image']
        d = float(row['distance'])
        new = int(row['bytes_new_gate_off'])
        if (img, d) not in baseline:
            continue
        old, cjxl = baseline[(img, d)]
        delta = new - old
        delta_pct = 100.0 * delta / old
        new_vs_cjxl_pct = 100.0 * (new - cjxl) / cjxl
        was_open = abs(100.0 * (old - cjxl) / cjxl) > 1.5
        is_open = abs(new_vs_cjxl_pct) > 1.5
        if was_open and not is_open:
            flips_to_fixed += 1
        if not was_open and is_open:
            new_open += 1
        if was_open:
            n_open += 1
            total_open_delta += delta
        total_delta += delta
        marker = ' [OPEN→FIXED]' if (was_open and not is_open) else (' [FIXED→OPEN]' if (not was_open and is_open) else (' [OPEN]' if is_open else ''))
        print(f"{img:<10} {d:<5} {old:>8} {new:>8} {delta:+8d} {delta_pct:>+9.3f}% {cjxl:>8} {new_vs_cjxl_pct:>+10.3f}%{marker}")

print(f"\nTotal delta: {total_delta:+d} B")
print(f"Open cells delta (n={n_open}): {total_open_delta:+d} B")
print(f"OPEN→FIXED flips: {flips_to_fixed}")
print(f"FIXED→OPEN regressions: {new_open}")
