#!/usr/bin/env python3
"""RD-efficiency of the qf-seed main-band lift: at MATCHED quality (ssim2),
how many bytes does cjxl need vs our lift_on? efficiency = cjxl_bytes/lift_on_bytes.
>1 => lift is more efficient than cjxl (genuine win); ~1 => on cjxl's curve
(neutral, just overshoots the target); <1 => lift is RD-dominated."""
import subprocess, os, glob, re, sys

CLI = "./target/release/cjxl-rs"
CJXL = os.path.expanduser("~/work/jxl-efforts/libjxl/build/tools/cjxl")
SCORE = glob.glob("target/release/examples/score_jxl_files")[0]
TMP = "/home/lilith/.claude/jobs/c81743c2/tmp"
GB = os.path.expanduser("~/work/codec-corpus/gb82-sc")
W = os.path.expanduser("~/work/codec-corpus/imazen-26/8100-lilith-web-screenshots")
IMZ = os.path.expanduser("~/work/codec-corpus/imazen-26/9226-lilith-ai-products/beauty")

def g(p):
    m = glob.glob(p)
    return m[0] if m else None

# (label, path, m3, class)
CELLS = [
    ("imac_dark",  f"{GB}/imac_dark.png", 21.0, "LOW-ship"),
    ("graph",      f"{GB}/graph.png",     11.8, "LOW-ship"),
    ("imac_g3",    f"{GB}/imac_g3.png",   14.3, "LOW-ship"),
    ("gmessages",  f"{GB}/gmessages.png", 10.2, "LOW-ship"),
    ("windows95",  f"{GB}/windows95.png", 27.2, "HIGH"),
    ("imessage",   f"{GB}/imessage.png",  67.6, "HIGH"),
    ("ing9291",    g(f"{IMZ}/9291_*.png"),28.2, "HIGH"),
    ("w8102_1440", g(f"{W}/1440x900/8102_*.png"),  26.9, "HIGH"),
    ("w8160_1920", g(f"{W}/1920x1080/8160_*.png"), 32.7, "HIGH"),
    ("w8221_2880", g(f"{W}/2880x1800/8221_*.png"), 26.8, "HIGH"),
    ("w8330_768",  g(f"{W}/768x1024/8330_*.png"),  39.2, "HIGH"),
    ("w8274_375",  g(f"{W}/375x667/8274_*.png"),   25.6, "HIGH"),
]
CJXL_DS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]

def enc_cli(src, out, d, env=None):
    e = dict(os.environ);
    if env: e.update(env)
    subprocess.run([CLI, "-e", "7", "-d", str(d), src, out], env=e,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return os.path.getsize(out) if os.path.exists(out) else 0

def enc_cjxl(src, out, d):
    subprocess.run([CJXL, src, out, "-d", str(d), "-e", "7"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return os.path.getsize(out) if os.path.exists(out) else 0

def score(src, arms):  # arms: list of (label, path) -> {label: ssim2}
    args = [SCORE, src] + [f"{l}={p}" for l, p in arms]
    out = subprocess.run(args, capture_output=True, text=True).stdout
    res = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4:
            m = re.match(r"(\w[\w.]*)", parts[0])
            try:
                res[parts[0]] = float(parts[-1])  # ssim2 last col
            except ValueError:
                pass
    return res

def interp_cjxl_bytes(curve, target_ssim2):
    # curve: list of (bytes, ssim2) sorted by ssim2 ascending; return bytes at target
    pts = sorted(curve, key=lambda x: x[1])
    if target_ssim2 <= pts[0][1]: return pts[0][0]
    if target_ssim2 >= pts[-1][1]: return pts[-1][0]
    for i in range(len(pts)-1):
        b0, s0 = pts[i]; b1, s1 = pts[i+1]
        if s0 <= target_ssim2 <= s1:
            if s1 == s0: return b0
            f = (target_ssim2 - s0) / (s1 - s0)
            return b0 + f * (b1 - b0)
    return pts[-1][0]

print(f"{'label':<12} {'m3':>5} {'class':<8} {'lift_on_B':>10} {'ssim2_on':>8} {'cjxl_B@q':>10} {'efficiency':>10} {'verdict'}")
rows = []
for label, path, m3, cls in CELLS:
    if not path or not os.path.exists(path):
        print(f"{label:<12} MISSING"); continue
    on = enc_cli(path, f"{TMP}/on.jxl", 4.0)
    arms = [("liftON", f"{TMP}/on.jxl")]
    curve = []
    for d in CJXL_DS:
        b = enc_cjxl(path, f"{TMP}/cj_{d}.jxl", d)
        arms.append((f"cj{str(d).replace('.','_')}", f"{TMP}/cj_{d}.jxl"))
    sc = score(path, arms)
    ssim2_on = sc.get("liftON", 0.0)
    for d in CJXL_DS:
        key = f"cj{str(d).replace('.','_')}"
        b = os.path.getsize(f"{TMP}/cj_{d}.jxl")
        if key in sc: curve.append((b, sc[key]))
    cj_b = interp_cjxl_bytes(curve, ssim2_on) if curve else 0
    eff = cj_b / on if on else 0
    verdict = "WIN" if eff >= 1.25 else ("neutral" if eff >= 0.95 else "LOSS")
    print(f"{label:<12} {m3:>5} {cls:<8} {on:>10} {ssim2_on:>8.2f} {cj_b:>10.0f} {eff:>9.2f}x {verdict}")
    rows.append((label, m3, cls, on, ssim2_on, cj_b, eff, verdict))

# summary by class
print("\n=== summary: mean efficiency by class ===")
for cls in ["LOW-ship", "HIGH"]:
    effs = [r[6] for r in rows if r[2] == cls]
    if effs:
        print(f"  {cls:<9} n={len(effs)} mean_eff={sum(effs)/len(effs):.2f}x  min={min(effs):.2f}x max={max(effs):.2f}x")

# write TSV
with open(f"{TMP}/rd_efficiency.tsv", "w") as f:
    f.write("label\tm3\tclass\tlift_on_bytes\tssim2_on\tcjxl_bytes_at_matched_ssim2\tefficiency\tverdict\n")
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
print(f"\nwrote {TMP}/rd_efficiency.tsv")
