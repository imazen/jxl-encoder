import os, subprocess, random, glob, shutil
random.seed(11)
REC="/home/lilith/work/zen/jxl-encoder/target/release/examples/jpeg_recompress"
ZJR="/home/lilith/work/zen/zenjpeg-recompress/target/release/examples/jxl_phase0_recompress"
OURS="/home/lilith/work/zen/jxl-encoder/target/release/cjxl-rs"
JXLOX="/home/lilith/work/jxl-efforts/jxl-oxide/target/release/jxl-oxide"
SSIM2="/home/lilith/work/jxl-efforts/libjxl/build/tools/ssimulacra2"
TMP="/tmp/rd"; shutil.rmtree(TMP,ignore_errors=True); os.makedirs(TMP)

# corpus
paths=[]
for d in ['/home/lilith/product-images','/home/lilith/work/codec-corpus']:
    r=subprocess.run(['find',d,'-type','f','-name','*.jpg'],capture_output=True,text=True,timeout=60); paths+=r.stdout.split('\n')
paths=[p for p in paths if p]; random.shuffle(paths)
srcs=[]
for p in paths:
    if len(srcs)>=12: break
    try: fr=subprocess.run(['file',p],capture_output=True,text=True,timeout=2)
    except: continue
    if 'components 3' in fr.stdout and ('baseline' in fr.stdout or 'progressive' in fr.stdout) and 'precision 8' in fr.stdout:
        srcs.append(p)

def decode_png(jxl,png):
    r=subprocess.run([JXLOX,'decode',jxl,'-o',png],capture_output=True,timeout=120)
    return r.returncode==0 and os.path.exists(png)
def ssim2(a,b):
    r=subprocess.run([SSIM2,a,b],capture_output=True,text=True,timeout=120)
    try: return float(r.stdout.strip().split('\n')[0])
    except: return None
def transcode(jpg,jxl):
    subprocess.run([OURS,'-e','7',jpg,jxl],capture_output=True,timeout=120)
    return os.path.getsize(jxl) if os.path.exists(jxl) else None

# per-file curves: list of (ssim2, size) for each method
methods={'PreserveJxl':[], 'zenjpeg_JPEG':[], 'pipeline_JXL':[]}
# store all points then interpolate
import collections
allpts={'PreserveJxl':[], 'zenjpeg_JPEG':[], 'pipeline_JXL':[]}
for i,src in enumerate(srcs):
    w=f"{TMP}/{i}"; os.makedirs(w,exist_ok=True)
    # reference = lossless transcode decoded via jxl-oxide
    REC and subprocess.run([REC,src,"1.0",f"{w}/ref.jxl"],capture_output=True,timeout=120)
    if not decode_png(f"{w}/ref.jxl",f"{w}/ref.png"): continue
    ref=f"{w}/ref.png"
    # PreserveJxl sweep
    pj=[]
    for s in [1.1,1.25,1.4,1.6,1.8,2.0,2.25,2.5,3.0,4.0]:
        o=f"{w}/pj_{s}.jxl"
        subprocess.run([REC,src,str(s),o],capture_output=True,timeout=120)
        if not os.path.exists(o): continue
        if not decode_png(o,f"{w}/pj_{s}.png"): continue
        q=ssim2(ref,f"{w}/pj_{s}.png")
        if q is not None: pj.append((q,os.path.getsize(o)))
    allpts['PreserveJxl'].append((src,pj))
    # zenjpeg targets
    zj=[]; pp=[]
    for t in [80,70,60,50,40,30,20]:
        td=f"{w}/zj_{t}"; os.makedirs(td,exist_ok=True)
        subprocess.run([ZJR,td,str(t),src],capture_output=True,timeout=120)
        jpg=glob.glob(f"{td}/*.jpg")
        if not jpg: continue
        jpg=jpg[0]; jsz=os.path.getsize(jpg)
        # transcode to jxl for consistent-decoder ssim2
        tx=f"{w}/zjx_{t}.jxl"
        txsz=transcode(jpg,tx)
        if txsz is None or not decode_png(tx,f"{w}/zjx_{t}.png"): continue
        q=ssim2(ref,f"{w}/zjx_{t}.png")
        if q is None: continue
        zj.append((q,jsz)); pp.append((q,txsz))
    allpts['zenjpeg_JPEG'].append((src,zj))
    allpts['pipeline_JXL'].append((src,pp))
    print(f"  file {i}: pj={len(pj)} zj={len(zj)} pts", flush=True)

# interpolate each method's per-file curve to target ssim2, sum sizes over files where reachable
def interp(curve, target):
    # curve: list of (ssim2, size); want smallest size achieving ssim2>=target (Pareto: lower envelope)
    pts=sorted(curve)  # by ssim2 asc
    cand=[sz for (q,sz) in pts if q>=target]
    return min(cand) if cand else None

print("\nssim2  method        files  total_bytes")
for tgt in [88,85,82,79,76,73,70]:
    row={}
    for m in methods:
        tot=0; n=0
        for (src,curve) in allpts[m]:
            sz=interp(curve,tgt)
            if sz is not None: tot+=sz; n+=1
        row[m]=(tot,n)
    # only compare files reachable by ALL methods
    common=[]
    for idx in range(len(allpts['PreserveJxl'])):
        ok=all(interp(allpts[m][idx][1],tgt) is not None for m in methods)
        if ok: common.append(idx)
    if not common: 
        print(f"{tgt:>5}  (no file reachable by all methods)"); continue
    sums={m:sum(interp(allpts[m][idx][1],tgt) for idx in common) for m in methods}
    base=sums['zenjpeg_JPEG']
    print(f"{tgt:>5}  n_common={len(common)}")
    for m in ['zenjpeg_JPEG','pipeline_JXL','PreserveJxl']:
        print(f"       {m:<14} {sums[m]:>11,d}  {sums[m]/base*100:>7.1f}% of zenjpeg-JPEG")
