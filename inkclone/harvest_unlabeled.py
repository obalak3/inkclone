"""Harvest symbol, digit and punctuation instances from the unlabeled stroke corpus by
shape-matching against clean seeds.

Single-stroke classes: every unlabeled stroke that (a) is isolated (no other stroke
overlaps its box much), (b) is nearest to a symbol seed rather than any letter seed,
and (c) is within a tight distance of that seed, becomes an instance. Two-stroke
classes (= + ≤ ≥ → ×) are matched as pairs of nearby strokes.

Harvested instances inherit the matched seed's size and baseline position, so they
drop into the symbol bank's coordinate system directly.
"""
import glob, json, pickle, collections
import numpy as np
from .units import resample

NP = 16
SINGLE = set("()[]{}|,./-<>0123567891'")
PAIRS = set("=+≤≥→×")


def feat(pieces_xy, size_ref):
    """Shape feature for a (list of) stroke(s): per-stroke resampled normalised polyline + size terms."""
    allp = np.vstack(pieces_xy); lo, hi = allp.min(0), allp.max(0); ext = max(hi - lo); ext = ext if ext > 1e-6 else 1.0
    parts = [((resample(p, NP) - lo) / ext).ravel() for p in pieces_xy]
    rel = (hi - lo) / size_ref
    return np.r_[np.concatenate(parts), rel * 2.0, (hi[0]-lo[0]) / max(hi[1]-lo[1], 1e-6) * 0.5]


def page_scale(strokes):
    h = np.array([s[:, 1].max() - s[:, 1].min() for s in strokes]); h = h[(h > 1.5) & (h < 25)]
    return float(np.median(h)) if len(h) else 8.0


def seeds_from_banks(sym, letters, n_strokes):
    """-> dict class -> list of (feature, pieces_norm) for seeds with exactly n_strokes main strokes."""
    S = collections.defaultdict(list)
    import json, os
    dseeds = {c: {tuple(k[:4]) for k in v} for c, v in json.load(open("data/curation/digit_seeds.json")).items()} if os.path.exists("data/curation/digit_seeds.json") else {}
    for c, insts in sym.items():
        for u in insts:
            if c.isdigit() and dseeds and tuple(u["src"]) not in dseeds.get(c, set()): continue
            if len(u["pieces"]) == n_strokes:  # cut-out brackets/commas are acceptable seeds; the grid showed them mostly clean
                xy = [p[:, :2] for p in u["pieces"]]; S[c].append((feat(xy, 1.3), u))  # bank units: xh=1 -> size_ref 1.3 (letter height)
    for c, insts in letters.items():
        if not c.isalpha(): continue
        for u in insts[:40]:
            if len(u["pieces"]) == n_strokes and u.get("whole"):
                S["L:" + c].append((feat([p[:, :2] for p in u["pieces"]], 1.3), u))
    return S


def clean_seeds(S):
    """Drop a seed when it is nearer to some other class's seeds than to any seed of its own class."""
    out = {}
    for c, lst in S.items():
        keep = []
        for k, (f, u) in enumerate(lst):
            own = [np.linalg.norm(f - g) / np.sqrt(len(f)) for j, (g, _) in enumerate(lst) if j != k]
            other = [np.linalg.norm(f - g) / np.sqrt(len(f)) for cc, l2 in S.items() if cc != c for g, _ in l2]
            if not own or not other or min(own) <= min(other) + 0.02: keep.append((f, u))
        if not keep and len(lst) >= 2 and not c.startswith("L:"):
            # keep the two most mutually similar seeds rather than losing the class
            best = min(((np.linalg.norm(a[0] - b[0]), i, j) for i, a in enumerate(lst) for j, b in enumerate(lst) if i < j))
            keep = [lst[best[1]], lst[best[2]]]
        if keep: out[c] = keep
    return out


def isolated(i, boxes, frac=0.25):
    b = boxes[i]; area = max((b[2]-b[0]) * (b[3]-b[1]), 1e-6)
    ix = np.minimum(boxes[:, 2], b[2]) - np.maximum(boxes[:, 0], b[0]); iy = np.minimum(boxes[:, 3], b[3]) - np.maximum(boxes[:, 1], b[1])
    inter = np.clip(ix, 0, None) * np.clip(iy, 0, None); inter[i] = 0
    return inter.max() / area < frac


def harvest(prefixes, sym, letters, thr1=0.20, thr2=0.24, margin=0.04, max_per_class=150):
    out = collections.defaultdict(list)
    S1 = clean_seeds(seeds_from_banks(sym, letters, 1)); S2 = clean_seeds(seeds_from_banks(sym, letters, 2))
    print("seeds after cleaning:", {c: len(v) for c, v in S1.items() if not c.startswith("L:")}, "| pairs:", {c: len(v) for c, v in S2.items() if not c.startswith("L:")})
    def classify(S, f):
        best = []
        for c, lst in S.items():
            d = min(np.linalg.norm(f - sf) / np.sqrt(len(f)) for sf, _ in lst); best.append((d, c))
        best.sort(); return best
    for prefix in prefixes:
        arrs = np.load(prefix + ".npz"); meta = json.load(open(prefix + ".json"))
        for pi, m in enumerate(meta):
            strokes = [arrs[f"p{pi}_s{si}"] for si in range(m["n_strokes"])]
            if len(strokes) < 5: continue
            ref = page_scale(strokes); boxes = np.array([[s[:, 0].min(), s[:, 1].min(), s[:, 0].max(), s[:, 1].max()] for s in strokes])
            for i, s in enumerate(strokes):
                h = boxes[i, 3] - boxes[i, 1]; w = boxes[i, 2] - boxes[i, 0]
                if max(h, w) < 0.08 * ref or max(h, w) > 3.5 * ref or len(s) < 4: continue
                if not isolated(i, boxes): continue
                best = classify(S1, feat([s[:, :2]], ref))
                d, c = best[0]
                if c.startswith("L:") or c not in SINGLE or d > thr1: continue
                d2 = best[1][0] if len(best) > 1 else 9  # runner-up of ANY class
                if d2 - d < margin and not (c == "0" and best[1][1] == "L:o"): continue
                out[c].append(adopt([s], S1[c], ref, prefix, pi, i))
            # pairs: consecutive strokes in pen order that are close
            for i in range(len(strokes) - 1):
                a, b = strokes[i], strokes[i+1]
                if abs(a[:, 1].mean() - b[:, 1].mean()) > 1.2 * ref or abs(a[:, 0].mean() - b[:, 0].mean()) > 1.5 * ref: continue
                if not (isolated(i, boxes, 0.6) and isolated(i+1, boxes, 0.6)): continue
                best = classify(S2, feat([a[:, :2], b[:, :2]], ref)); d, c = best[0]
                if c.startswith("L:") or c not in PAIRS or d > thr2: continue
                if len(best) > 1 and best[1][0] - d < margin: continue
                out[c].append(adopt([a, b], S2[c], ref, prefix, pi, i))
    for c in out:
        out[c].sort(key=lambda u: u["dist"]); out[c] = out[c][:max_per_class]
    return dict(out)


def adopt(strokes, seeds, ref, prefix, pi, i):
    """Map the candidate into bank coordinates using the nearest seed's size and position."""
    f = feat([s[:, :2] for s in strokes], ref)
    d, (sf, seed) = min((np.linalg.norm(f - sf) / np.sqrt(len(f)), (sf, seed)) for sf, seed in seeds)
    sp = np.vstack([p[:, :2] for p in seed["pieces"]]); cp = np.vstack([s[:, :2] for s in strokes])
    s_lo, s_hi = sp.min(0), sp.max(0); c_lo, c_hi = cp.min(0), cp.max(0)
    scale = max(s_hi - s_lo) / max(max(c_hi - c_lo), 1e-6)
    pieces = [np.column_stack([(s[:, 0] - c_lo[0]) * scale + 0.0, (s[:, 1] - c_lo[1]) * scale + s_lo[1], s[:, 2]]) for s in strokes]
    allq = np.vstack([p[:, :2] for p in pieces])
    return {"text": seed["text"], "pieces": pieces, "width": allq[:, 0].max(), "height": allq[:, 1].max() - allq[:, 1].min(),
            "src": (prefix, pi, i), "single": True, "harvested": True, "dist": float(d)}


if __name__ == "__main__":
    sym = pickle.load(open("data/derived/symbols.pkl", "rb")); L = pickle.load(open("data/derived/units.pkl", "rb"))["letters"]
    prefixes = [p[:-5] for p in sorted(glob.glob("data/derived/*.json")) if not p.endswith(".words.json")]
    H = harvest(prefixes, sym, L)
    # union with earlier harvests (keyed by source stroke) so a rebuild never loses a class it once had
    import os
    if os.path.exists("data/derived/symbols_harvested.pkl"):
        old = pickle.load(open("data/derived/symbols_harvested.pkl", "rb"))
        for c, insts in old.items():
            have = {tuple(u["src"]) for u in H.get(c, [])}
            H.setdefault(c, []).extend(u for u in insts if tuple(u["src"]) not in have)
            H[c].sort(key=lambda u: u["dist"]); H[c] = H[c][:150]
    print("harvested (union):", {c: len(v) for c, v in sorted(H.items())})
    pickle.dump(H, open("data/derived/symbols_harvested.pkl", "wb"))
