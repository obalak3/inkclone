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
                cu = unit(normalise(pieces, px, baseline, xh), chars, src); cu["run0"] = run[0]; C[chars].append(cu)
        for k, ch in enumerate(seg):
            pieces = [w["strokes"][i][i0:i1].copy() for i, i0, i1 in ch["pieces"]]
            px = min(p[:, 0].min() for p in pieces)
            u = unit(normalise(pieces, px, baseline, xh), ch["char"], src)
            u["whole"] = not ch["joined_prev"] and not ch["joined_next"]  # complete pen stroke(s), no guessed cut
            lo, hi = HEIGHT_RANGE[letter_class(ch["char"])]
            if lo <= u["height"] <= hi and 0.15 <= u["width"] <= 4.5: L[ch["char"]].append(u)
    # whole-word outlier filter: for words seen 3+ times, drop instances whose width or
    # stroke count is far from that word's median (stray strokes, swallowed neighbours)
    wrep = [0, 0]
    for t, insts in list(W.items()):
        wrep[0] += len(insts)
        if len(insts) >= 3:
            wd = np.median([i["width"] for i in insts]); ns = np.median([len(i["pieces"]) for i in insts])
            W[t] = [i for i in insts if abs(i["width"] - wd) <= 0.35 * wd and abs(len(i["pieces"]) - ns) <= 2]
        wrep[1] += len(W[t])
    # shape filter on letters
    report = {}
    for c, insts in L.items():
        if len(insts) < min_for_filter: report[c] = (len(insts), len(insts)); continue
        X = np.array([shape_vec(i["pieces"]) for i in insts]); X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        D = np.linalg.norm(X[:, None] - X[None], axis=2); np.fill_diagonal(D, np.inf)
        score = np.sort(D, 1)[:, :5].mean(1); keep = score <= np.quantile(score, 1 - drop_frac)
        L[c] = [i for i, k in zip(insts, keep) if k]; report[c] = (len(insts), int(keep.sum()))
    # template scores: how well does each letter instance / word instance match its label?
    from .recut import build_templates, NPTS
    T = build_templates(L)
    def inst_desc(pieces):
        main = max(pieces, key=len)[:, :2]; pts = np.vstack([p[:, :2] for p in pieces])
        r = resample(main - [main[:, 0].min(), 0], NPTS)
        return np.r_[r.ravel(), pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min()]
    wts = np.ones(2 * NPTS + 2); wts[-2:] = 3.0
    def score(c, pieces):
        if c not in T: return None
        d = np.sqrt((((inst_desc(pieces) - T[c]) ** 2) * wts).sum(1) / len(wts)); d.sort()
        return float(d[1] if len(d) > 1 else d[0])  # skip self-match
    for c, insts in L.items():
        for u in insts: u["score"] = score(c, u["pieces"])
    # word instances: mean letter score over the word's segmentation
    word_scores = []; chunk_scores = {}
    for w, seg in zip(words, segs):
        if not seg: continue
        fr = word_frame(w["strokes"], seg)
        if not fr: continue
        baseline, xh = fr; sc = []
        for ch in seg:
            pieces = [w["strokes"][i][i0:i1].copy() for i, i0, i1 in ch["pieces"]]
            px = min(p[:, 0].min() for p in pieces); s_ = score(ch["char"], normalise(pieces, px, baseline, xh))
            if s_ is not None: sc.append(s_)
        word_scores.append((w["text"], (w["source"], w["page"], w["text"]), float(np.mean(sc)) if sc else None))
        # per-run (chunk) scores for this word
        per_char = []
        for ch in seg:
            pieces = [w["strokes"][i][i0:i1].copy() for i, i0, i1 in ch["pieces"]]
            px = min(p[:, 0].min() for p in pieces); per_char.append(score(ch["char"], normalise(pieces, px, baseline, xh)))
        run = [0]
        for k in range(1, len(seg)):
            if seg[k]["joined_prev"]: run.append(k)
            else: run = [k]
            # store on every prefix end so lookup by (src, start, chars) works
        runs, run = [], [0]
        for k in range(1, len(seg)):
            if seg[k]["joined_prev"]: run.append(k)
            else: runs.append(run); run = [k]
        runs.append(run)
        for run in runs:
            if len(run) < 2: continue
            chars = "".join(seg[k]["char"] for k in run); scs = [per_char[k] for k in run if per_char[k] is not None]
            chunk_scores[((w["source"], w["page"], w["text"]), chars, run[0])] = float(np.mean(scs)) if scs else None
    by_src = {src: sc for _, src, sc in word_scores}
    vals = np.array([sc for _, _, sc in word_scores if sc is not None]); cut = float(np.quantile(vals, 0.85))
    dropped = 0
    for t, insts in list(W.items()):
        kept = []
        for u in insts:
            u["score"] = by_src.get(u["src"]); 
            if u["score"] is not None and u["score"] > cut: dropped += 1
            else: kept.append(u)
        if kept: W[t] = kept
        else: del W[t]
    cdrop = 0
    for t, insts in list(C.items()):
        kept = []
        for u in insts:
            u["score"] = chunk_scores.get((u["src"], t, u["run0"]))
            if u["score"] is not None and u["score"] > cut: cdrop += 1
            else: kept.append(u)
        if kept: C[t] = kept
        else: del C[t]
    report['_chunks'] = cdrop
    report['_words'] = tuple(wrep) + (dropped, cut)
    return dict(W), dict(C), dict(L), report


if __name__ == "__main__":
    d = pickle.load(open("data/derived/segments.pkl", "rb"))
    words = load_words(sorted({w["source"] for w in d["words"]}))
    W, C, L, rep = build(words, d["segs"])
    pickle.dump({"words": W, "chunks": C, "letters": L}, open("data/derived/units.pkl", "wb"))
    print(f"words: {sum(map(len, W.values()))} instances / {len(W)} distinct | chunks: {sum(map(len, C.values()))} / {len(C)} | letters kept: {sum(map(len, L.values()))}")
    print("chunks dropped for template mismatch:", rep.pop("_chunks"))
    wr = rep.pop("_words"); print(f"word filters: {wr[0]} -> {wr[1]} (outliers) -> {wr[1]-wr[2]} (template mismatch, cut {wr[3]:.2f})")
    nw = sum(1 for v in L.values() for i in v if i.get("whole")); print(f"letters that are complete strokes (no cut): {nw}/{sum(map(len, L.values()))}")
    print("capitals kept:", " ".join(f"{c}:{b}" for c, (a, b) in sorted(rep.items()) if c.isupper()))
