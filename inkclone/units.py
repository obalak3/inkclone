"""Unit banks for longest-unit synthesis: whole words, joined chunks, filtered letters.

All units are stored in word-normalised coordinates (baseline y=0, x-height = 1, x from
the unit's own left edge, pen width in pt). Letters pass a shape filter: for letters
with enough instances, the ones far from their nearest neighbours in a resampled-shape
space are dropped (these are the mis-cut instances carrying neighbour fragments).
"""
import pickle, collections
import numpy as np
from .segment import load_words
from .bank import word_frame, letter_class, HEIGHT_RANGE


def resample(pts, n=24):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    if d[-1] < 1e-6: return np.repeat(pts[:1], n, 0)
    t = np.linspace(0, d[-1], n); return np.column_stack([np.interp(t, d, pts[:, 0]), np.interp(t, d, pts[:, 1])])


def shape_vec(pieces):
    main = max(pieces, key=len)[:, :2]
    r = resample(main); allp = np.vstack([p[:, :2] for p in pieces])
    return np.r_[r.ravel(), allp[:, 0].max() - allp[:, 0].min(), allp[:, 1].max(), allp[:, 1].min(), len(pieces)]


def normalise(pieces, x0, baseline, xh):
    return [np.column_stack([(p[:, 0] - x0) / xh, (p[:, 1] - baseline) / xh, p[:, 2]]) for p in pieces]


def unit(pieces, text, src):
    allp = np.vstack([p[:, :2] for p in pieces])
    return {"text": text, "pieces": pieces, "width": allp[:, 0].max(), "height": allp[:, 1].max() - allp[:, 1].min(), "src": src}


def build(words, segs, drop_frac=0.3, min_for_filter=20):
    W, C, L = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    for w, seg in zip(words, segs):
        if not seg: continue
        fr = word_frame(w["strokes"], seg)
        if not fr: continue
        baseline, xh = fr; src = (w["source"], w["page"], w["text"])
        allp = np.vstack([s[:, :2] for s in w["strokes"]])
        # whole word
        x0 = allp[:, 0].min()
        W[w["text"]].append(unit(normalise([s.copy() for s in w["strokes"]], x0, baseline, xh), w["text"], src))
        # joined runs -> chunks; every char -> letter
        runs, run = [], [0]
        for k in range(1, len(seg)):
            if seg[k]["joined_prev"]: run.append(k)
            else: runs.append(run); run = [k]
        runs.append(run)
        for run in runs:
            chars = "".join(seg[k]["char"] for k in run)
            pieces = [w["strokes"][i][i0:i1].copy() for k in run for i, i0, i1 in seg[k]["pieces"]]
            if len(run) >= 2:
                px = min(p[:, 0].min() for p in pieces)
                C[chars].append(unit(normalise(pieces, px, baseline, xh), chars, src))
        for k, ch in enumerate(seg):
            pieces = [w["strokes"][i][i0:i1].copy() for i, i0, i1 in ch["pieces"]]
            px = min(p[:, 0].min() for p in pieces)
            u = unit(normalise(pieces, px, baseline, xh), ch["char"], src)
            lo, hi = HEIGHT_RANGE[letter_class(ch["char"])]
            if lo <= u["height"] <= hi and 0.15 <= u["width"] <= 3.5: L[ch["char"]].append(u)
    # shape filter on letters
    report = {}
    for c, insts in L.items():
        if len(insts) < min_for_filter: report[c] = (len(insts), len(insts)); continue
        X = np.array([shape_vec(i["pieces"]) for i in insts]); X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        D = np.linalg.norm(X[:, None] - X[None], axis=2); np.fill_diagonal(D, np.inf)
        score = np.sort(D, 1)[:, :5].mean(1); keep = score <= np.quantile(score, 1 - drop_frac)
        L[c] = [i for i, k in zip(insts, keep) if k]; report[c] = (len(insts), int(keep.sum()))
    return dict(W), dict(C), dict(L), report


if __name__ == "__main__":
    d = pickle.load(open("data/derived/segments.pkl", "rb"))
    words = load_words(sorted({w["source"] for w in d["words"]}))
    W, C, L, rep = build(words, d["segs"])
    pickle.dump({"words": W, "chunks": C, "letters": L}, open("data/derived/units.pkl", "wb"))
    print(f"words: {sum(map(len, W.values()))} instances / {len(W)} distinct | chunks: {sum(map(len, C.values()))} / {len(C)} | letters kept: {sum(map(len, L.values()))}")
    print("letter filter (before->after):", " ".join(f"{c}:{a}->{b}" for c, (a, b) in sorted(rep.items()) if c.islower()))
