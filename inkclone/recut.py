"""Template-guided re-segmentation.

Second pass over every labeled word. Same DP as `segment.py`, but each candidate
span is also scored by how much it looks like the letter it is supposed to be, using
templates built from the shape-filtered letter bank (the clean cores). Spans are
described by a resampled polyline in word-normalised coordinates (baseline y=0,
x-height = 1) plus width and height. This is what stops a cut from handing the tail
of an "h" to the following "e".
"""
import pickle
import numpy as np
from .segment import split_diacritics, default_width_prior, load_words
from .bank import word_frame
from .units import resample

NPTS = 16


def span_desc(pts, xh):
    """pts (n,2) in page coords relative to baseline -> descriptor in xh units."""
    q = pts / xh; q = q - [q[:, 0].min(), 0]
    r = resample(q, NPTS)
    return np.r_[r.ravel(), q[:, 0].max() - q[:, 0].min(), q[:, 1].max() - q[:, 1].min()]


def build_templates(letters, per_letter=60):
    T = {}
    for c, insts in letters.items():
        descs = []
        for inst in insts[:per_letter]:
            main = max(inst["pieces"], key=len)[:, :2]  # already normalised: baseline 0, xh 1
            pts = np.vstack([p[:, :2] for p in inst["pieces"]])
            r = resample(main - [main[:, 0].min(), 0], NPTS)
            descs.append(np.r_[r.ravel(), pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min()])
        if descs: T[c] = np.array(descs)
    return T


def recut_word(strokes, text, prior, templates, baseline, xh, lam_join=0.6, lam_width=0.6, lam_shape=1.2, max_b=44):
    K = len(text)
    allpts = np.vstack([s[:, :2] for s in strokes]); word_h = allpts[:, 1].max() - allpts[:, 1].min()
    main, small = split_diacritics(strokes, word_h)
    if not main or K == 1: return None
    seq = np.vstack([s[:, :2] for _, s in main]); N = len(seq)
    starts = np.cumsum([0] + [len(s) for _, s in main]); penup = set(starts[1:-1].tolist())
    step = max(1, N // max_b)
    cands = sorted(set(range(step, N - step, step)) | penup); cands = [c for c in cands if 0 < c < N]
    if len(cands) < K - 1: return None
    bounds_all = [0] + cands + [N]; B = len(bounds_all); barr = np.array(bounds_all)
    pw = np.array([prior.get(c, 1.0) for c in text]); unit = (allpts[:, 0].max() - allpts[:, 0].min()) / pw.sum(); exp_w = pw * unit
    ybot = np.percentile(seq[:, 1], 85)
    jc = np.array([0.0 if b in penup or b in (0, N) else lam_join * (1 + max(0.0, (ybot - seq[b, 1]) / word_h)) for b in bounds_all])
    rel = (seq - [0, baseline]) / xh
    max_span = min(B - 1, int(np.ceil(1.6 * B / K)) + 2)
    # span width matrix (B x B) via running max/min from each start
    W = np.full((B, B), np.inf)
    for i in range(B):
        a = bounds_all[i]; xs = seq[a:, 0]
        if len(xs) == 0: continue
        cmax = np.maximum.accumulate(xs); cmin = np.minimum.accumulate(xs)
        js = np.arange(i + 1, min(B, i + max_span + 1)); ends = barr[js] - a - 1
        W[i, js] = cmax[ends] - cmin[ends]
    # descriptors for all allowed spans
    pairs = [(i, j) for i in range(B) for j in range(i + 1, min(B, i + max_span + 1)) if bounds_all[j] - bounds_all[i] >= 3]
    desc = np.empty((len(pairs), 2 * NPTS + 2))
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(rel, axis=0), axis=1))]
    for n, (i, j) in enumerate(pairs):
        a, b = bounds_all[i], bounds_all[j]; q = rel[a:b]
        L = cum[b-1] - cum[a]
        if L < 1e-6: r = np.repeat(q[:1], NPTS, 0)
        else:
            t = np.linspace(cum[a], cum[b-1], NPTS); r = np.column_stack([np.interp(t, cum[a:b], q[:, 0]), np.interp(t, cum[a:b], q[:, 1])])
        r = r - [q[:, 0].min(), 0]
        desc[n, :2*NPTS] = r.ravel(); desc[n, -2] = q[:, 0].max() - q[:, 0].min(); desc[n, -1] = q[:, 1].max() - q[:, 1].min()
    wts = np.ones(desc.shape[1]); wts[-2:] = 3.0
    shape = np.full((len(pairs), K), 5.0)
    for k, c in enumerate(text):
        T = templates.get(c)
        if T is None: continue
        d2 = ((desc[:, None, :] - T[None]) ** 2 * wts).sum(2)
        shape[:, k] = np.sqrt(d2.min(1) / desc.shape[1])
    # full cost tensor (B x B x K)
    C = np.full((B, B, K), np.inf)
    for n, (i, j) in enumerate(pairs):
        C[i, j, :] = lam_shape * shape[n]
    wc = lam_width * ((W[:, :, None] - exp_w[None, None, :]) / (0.5 * exp_w + 0.3 * unit)[None, None, :]) ** 2
    C = C + wc + jc[None, :, None]
    INF = 1e18; best = np.full((K, B), INF); back = np.full((K, B), -1, int)
    best[0, :] = C[0, :, 0]
    for k in range(1, K):
        js = range(k + 1, B) if k < K - 1 else [B - 1]
        for j in js:
            v = best[k-1, :j] + C[:j, j, k]
            i = int(np.argmin(v)); best[k, j] = v[i]; back[k, j] = i
    if not np.isfinite(best[K-1, B-1]): return None
    j = B - 1; cuts = []
    for k in range(K - 1, 0, -1):
        j = back[k, j]; cuts.append(bounds_all[j])
    bounds = [0] + cuts[::-1] + [N]
    chars = []
    for k in range(K):
        a, b = bounds[k], bounds[k+1]; pieces = []
        for m, (i, s) in enumerate(main):
            s0 = starts[m]; lo, hi = max(a, s0), min(b, s0 + len(s))
            if hi > lo: pieces.append((i, lo - s0, hi - s0))
        chars.append({"char": text[k], "pieces": pieces, "joined_prev": k > 0 and bounds[k] not in penup,
                      "joined_next": k < K - 1 and bounds[k+1] not in penup})
    spans = []
    for ch in chars:
        xs = np.concatenate([strokes[i][i0:i1, 0] for i, i0, i1 in ch["pieces"]]) if ch["pieces"] else np.array([np.nan])
        spans.append((np.nanmin(xs), np.nanmax(xs)))
    for i, s in small:
        cx = s[:, 0].mean(); d = [0 if lo <= cx <= hi else min(abs(cx - lo), abs(cx - hi)) for lo, hi in spans]
        chars[int(np.argmin(d))]["pieces"].append((i, 0, len(s)))
    return chars


if __name__ == "__main__":
    import time
    d = pickle.load(open("data/derived/segments.pkl", "rb")); units = pickle.load(open("data/derived/units.pkl", "rb"))
    words = load_words(sorted({w["source"] for w in d["words"]}))
    templates = build_templates(units["letters"]); prior = d["prior"]
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    t = time.time(); new = []; changed = 0
    for n, (w, seg) in enumerate(zip(words, d["segs"])):
        if limit and n >= limit: new.append(seg); continue
        fr = word_frame(w["strokes"], seg) if seg else None
        if not fr: new.append(seg); continue
        r = recut_word(w["strokes"], w["text"], prior, templates, *fr)
        if r is None: new.append(seg); continue
        if any(a["pieces"] != b["pieces"] for a, b in zip(r, seg)): changed += 1
        new.append(r)
    print(f"recut {len(words)} words in {time.time()-t:.0f}s, {changed} changed")
    if not limit: d["segs"] = new; pickle.dump(d, open("data/derived/segments.pkl", "wb"))
