# -*- coding: utf-8 -*-
"""
Audit v5: single candidate list
- Candidate pairs produced by all signals (SHA / pHash / E3 / E4) are merged into one CSV
- Sorted by confidence; judge each pair manually and fill in the `same` column
- Montage filenames carry a confidence prefix, so you can review them in order
Output:
  candidates.csv            Candidate pair list (for you to fill in `same`)
  candidates_vis/<class>/   Montages
"""
import os, csv, hashlib, re
from collections import defaultdict
from datetime import datetime
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ExifTags
import imagehash

DATASET_PATH = "/openbayes/input/input0"
CACHE_NPZ    = "embeddings_dinov2_vits14_orig1091.npz"
OUT_CSV      = "candidates.csv"
OUT_VIS      = "candidates_vis"

PHASH_THRESHOLD = 14
THIGH = 0.97
TLOW  = 0.92
TIME_GAP_SEC = 20
VALID_EXTS = {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}
H, PAD, BAR = 400, 12, 46
os.makedirs(OUT_VIS, exist_ok=True)

# Confidence levels (lower number = more reliable)
CONF = {
    "SHA":             1,
    "pHash=0":         2,
    "pHash_1_7":       3,
    "pHash_8_13":      4,
    "E3_high":         4,   # cos >= 0.99
    "E3":              5,   # 0.97 <= cos < 0.99
    "E4":              6,
}

def build_index(root):
    m = {}
    for r,_,fs in os.walk(root):
        for f in fs:
            if os.path.splitext(f)[1].lower() in VALID_EXTS:
                p = os.path.join(r,f)
                m[os.path.relpath(p,root).replace(os.sep,"/")] = p
    return m

def sha256_of(p):
    h = hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(1<<20), b""): h.update(b)
    return h.hexdigest()

def phash_bits(p):
    with Image.open(p) as im:
        im = ImageOps.exif_transpose(im)
        return np.asarray(imagehash.phash(im, hash_size=8).hash, np.uint8).ravel()

def exif_time(p):
    try:
        with Image.open(p) as im:
            ex = im.getexif()
            for tid, name in ExifTags.TAGS.items():
                if name == "DateTimeOriginal" and tid in ex: return str(ex[tid])
            if 306 in ex: return str(ex[306])
    except Exception: pass
    return None

def contiguous(ka, kb, name2path):
    na, nb = os.path.basename(ka), os.path.basename(kb)
    a = re.match(r"^(.*?)(\d+)(\D*)$", os.path.splitext(na)[0])
    b = re.match(r"^(.*?)(\d+)(\D*)$", os.path.splitext(nb)[0])
    if a and b and a.group(1)==b.group(1) and a.group(3)==b.group(3):
        if abs(int(a.group(2))-int(b.group(2))) == 1: return True
    ta, tb = exif_time(name2path[ka]), exif_time(name2path[kb])
    if ta and tb:
        try:
            fa = datetime.strptime(ta[:19], "%Y:%m:%d %H:%M:%S")
            fb = datetime.strptime(tb[:19], "%Y:%m:%d %H:%M:%S")
            return abs((fa-fb).total_seconds()) <= TIME_GAP_SEC
        except Exception: pass
    return False

def load_img(p, h=H):
    im = Image.open(p); im = ImageOps.exif_transpose(im).convert("RGB")
    w, hh = im.size
    return im.resize((int(w*h/hh), h), Image.LANCZOS)

def main():
    name2path = build_index(DATASET_PATH)
    keys = sorted(name2path.keys()); n = len(keys)
    print(f"[i] Total: {n}")

    # Image attributes
    sha_map = {}
    for k in keys:
        try: sha_map[k] = sha256_of(name2path[k])
        except Exception as e: print(f"[warn] sha {k}: {e}")

    bits = {}
    for k in keys:
        try: bits[k] = phash_bits(name2path[k])
        except Exception as e: print(f"[warn] phash {k}: {e}")

    emb = None
    if os.path.exists(CACHE_NPZ):
        d = np.load(CACHE_NPZ, allow_pickle=True)
        emb = {d["names"][i]: d["emb"][i] for i in range(len(d["names"]))}
        print(f"[i] Loaded embeddings for {len(emb)} images")

    cls2keys = defaultdict(list)
    for k in keys: cls2keys[k.split("/")[0]].append(k)

    # Collect candidates (pair -> best signal)
    cand = {}     # key: (cls, file_a, file_b) -> dict
    def add(ka, kb, signal, conf, phash, cos):
        if ka.split("/")[0] != kb.split("/")[0]: return
        fa, fb = os.path.basename(ka), os.path.basename(kb)
        if fa > fb: ka, kb, fa, fb = kb, ka, fb, fa
        key = (ka.split("/")[0], fa, fb)
        if key in cand and cand[key]["confidence"] <= conf:
            return
        cand[key] = dict(class_=key[0], file_a=fa, file_b=fb,
                         signal=signal, confidence=conf,
                         phash=phash, cos_sim=cos, same="")

    # SHA
    seen = defaultdict(list)
    for k in keys: seen[sha_map[k]].append(k)
    for h, g in seen.items():
        if len(g) > 1:
            g = sorted(g)
            for i in range(1, len(g)):
                pa = name2path[g[i-1]]; pb = name2path[g[i]]
                add(g[i-1], g[i], "SHA", CONF["SHA"], 0, 1.0)

    # pHash
    ks = [k for k in keys if k in bits]
    for a in range(len(ks)):
        for b in range(a+1, len(ks)):
            ka, kb = ks[a], ks[b]
            d = int(np.bitwise_xor(bits[ka], bits[kb]).sum())
            if d < PHASH_THRESHOLD:
                if   d == 0: sig, cf = "pHash=0",    CONF["pHash=0"]
                elif d <= 7: sig, cf = "pHash_1_7",  CONF["pHash_1_7"]
                else:        sig, cf = "pHash_8_13", CONF["pHash_8_13"]
                cos = float(emb[ka] @ emb[kb]) if (emb and ka in emb and kb in emb) else -1
                add(ka, kb, sig, cf, d, round(cos,4))

    # E3 / E4
    if emb:
        for cls, kk in cls2keys.items():
            for a in range(len(kk)):
                for b in range(a+1, len(kk)):
                    ka, kb = kk[a], kk[b]
                    if not (ka in emb and kb in emb): continue
                    cos = float(emb[ka] @ emb[kb])
                    if cos >= 0.99:
                        add(ka, kb, "E3_high", CONF["E3_high"], -1, round(cos,4))
                    elif cos >= THIGH:
                        add(ka, kb, "E3", CONF["E3"], -1, round(cos,4))
                    elif cos >= TLOW and contiguous(ka, kb, name2path):
                        add(ka, kb, "E4", CONF["E4"], -1, round(cos,4))

    rows = sorted(cand.values(), key=lambda r: (r["confidence"], r["class_"], r["file_a"]))

    # Write CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(["class","file_a","file_b","signal","confidence","phash","cos_sim","same"])
        for r in rows:
            w.writerow([r["class_"], r["file_a"], r["file_b"], r["signal"],
                        r["confidence"], r["phash"], r["cos_sim"], ""])

    # Montages
    counts = defaultdict(int)
    for i, r in enumerate(rows, 1):
        cls = r["class_"]
        pa = os.path.join(DATASET_PATH, cls, r["file_a"])
        pb = os.path.join(DATASET_PATH, cls, r["file_b"])
        if not (os.path.exists(pa) and os.path.exists(pb)): continue
        try: ia, ib = load_img(pa), load_img(pb)
        except Exception: continue
        W = ia.width + ib.width + 3*PAD
        canvas = Image.new("RGB", (W, H+BAR+PAD), "white")
        canvas.paste(ia, (PAD, BAR)); canvas.paste(ib, (PAD*2+ia.width, BAR))
        d = ImageDraw.Draw(canvas)
        d.text((PAD, 6),  f"[{cls}] #{i:04d} {r['signal']:>10} conf={r['confidence']}  "
                          f"{r['file_a']}  |  {r['file_b']}", fill="black")
        d.text((PAD, 26), f"phash={r['phash']}  cos={r['cos_sim']}  "
                          f"->  same image? (y/n)", fill="darkred")
        folder = os.path.join(OUT_VIS, cls.replace(" ", "_"))
        os.makedirs(folder, exist_ok=True)
        canvas.save(os.path.join(folder,
                    f"{i:04d}_conf{r['confidence']}_{r['signal']}.jpg"), quality=88)
        counts[r["signal"]] += 1

    print(f"\n===== Candidate list (single) =====")
    print(f"Total {len(rows)} pairs")
    for sig in sorted(counts, key=lambda s: CONF.get(s, 99)):
        print(f"  {sig:>10} : {counts[sig]} pairs")
    print(f"\n-> {OUT_CSV}")
    print(f"-> Montage directory {OUT_VIS}/")
    print(f"   Montage indices are sorted by confidence (smaller = more reliable); "
          f"review them one by one and fill in the `same` column")

if __name__ == "__main__":
    main()