#!/usr/bin/env python3
"""MABSplit Phase-0 variance analysis (issue #64 CHUNK 2 Phase 0).

Consumes JXL_MABSPLIT_DUMP TSVs (one per encode cell; lines emitted by the
`__env_var_diagnostics` hook in modular/tree_learn.rs):

    weighted_total<TAB>base_bits<TAB>chosen_prop|-1<TAB>best_bits<TAB>p:tot,p:tot,...

For each cell it derives, per find_best_split node:
  gain_p      = base_bits - best_total_p   (per evaluated property)
  winner gain = max gain (== base_bits - best_bits when a split was chosen)
  margin      = g(1) - g(2)                (winner vs runner-up property)
  margin_rel  = margin / g(1)              (how decisively the winner wins)

The MAB / successive-halving question: can losing properties be pruned on
partial data? Phase-0 proxy: if margins are typically a large fraction of
the winner's gain, property ranking is decided early and pruning is
plausible; if margins are razor-thin, every property must be evaluated to
near-completion and a bandit saves little. (Full Hoeffding constants need
partial-sample reruns — Phase 1; this distribution is the go/no-go.)

Usage: mabsplit_phase0_analyze.py name=path.tsv [name=path.tsv ...]
Emits a markdown table per cell + an overall verdict block.
"""

import sys


def q(sorted_vals, frac):
    if not sorted_vals:
        return float("nan")
    i = min(int(frac * (len(sorted_vals) - 1)), len(sorted_vals) - 1)
    return sorted_vals[i]


def analyze(path):
    nodes = 0
    no_split = 0
    decided = []  # (weighted_total, winner_gain, margin, margin_rel, n_props)
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                continue
            nodes += 1
            wt = int(parts[0])
            base = float(parts[1])
            chosen = int(parts[2])
            pairs = []
            if parts[4]:
                for tok in parts[4].split(","):
                    pname, val = tok.split(":")
                    pairs.append((int(pname), float(val)))
            if chosen < 0 or len(pairs) == 0:
                no_split += 1
                continue
            gains = sorted((base - tot for _, tot in pairs), reverse=True)
            g1 = gains[0]
            if g1 <= 0:
                no_split += 1
                continue
            g2 = gains[1] if len(gains) > 1 else 0.0
            margin = g1 - max(g2, 0.0)
            decided.append((wt, g1, margin, margin / g1, len(pairs)))
    return nodes, no_split, decided


def main():
    print("| cell | nodes | no-split | med props | winner-gain p50 (bits) "
          "| margin_rel p25 | p50 | p75 | decisive (rel>0.5) |")
    print("|---|---|---|---|---|---|---|---|---|")
    all_rel = []
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        nodes, no_split, dec = analyze(path)
        rels = sorted(d[3] for d in dec)
        gains = sorted(d[1] for d in dec)
        nprops = sorted(d[4] for d in dec)
        decisive = sum(1 for r in rels if r > 0.5) / len(rels) if rels else 0
        all_rel.extend(rels)
        print(f"| {name} | {nodes} | {no_split} | {q(nprops, 0.5):.0f} "
              f"| {q(gains, 0.5):.0f} | {q(rels, 0.25):.3f} "
              f"| {q(rels, 0.5):.3f} | {q(rels, 0.75):.3f} | {decisive:.1%} |")
    all_rel.sort()
    print(f"\noverall margin_rel: p25={q(all_rel, 0.25):.3f} "
          f"p50={q(all_rel, 0.5):.3f} p75={q(all_rel, 0.75):.3f} "
          f"(n={len(all_rel)} decided nodes)")


if __name__ == "__main__":
    main()
