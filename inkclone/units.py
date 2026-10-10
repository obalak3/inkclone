"""Unit banks for longest-unit synthesis: whole words, joined chunks, filtered letters.

All units are stored in word-normalised coordinates (baseline y=0, x-height = 1, x from
the unit's own left edge, pen width in pt). Letters pass a shape filter: for letters
with enough instances, the ones far from their nearest neighbours in a resampled-shape
space are dropped (these are the mis-cut instances carrying neighbour fragments).
"""
import pickle, collections
import numpy as np
from .segment import load_words
from .bank import word_frame, letter_class, HEIGHT_RANGE, slant_deg


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
    return {"text": text, "pieces": pieces, "width": allp[:, 0].max(), "height": allp[:, 1].max() - allp[:, 1].min(), "src": src,
            "slant": slant_deg(allp), "pen": float(np.median(np.concatenate([p[:, 2] for p in pieces])))}


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
            u = unit(normalise(pieces, px, baseline, xh), ch["char"], src); u["k"] = k
            u["whole"] = not ch["joined_prev"] and not ch["joined_next"]  # complete pen stroke(s), no guessed cut
            lo, hi = HEIGHT_RANGE[letter_class(ch["char"])]
            if lo <= u["height"] <= hi and 0.15 <= u["width"] <= 4.5: L[ch["char"]].append(u)
    # ink-density filter: scribbled or over-written word instances carry far more ink per letter than normal
    dens = {}
    for t, insts in W.items():
        for u in insts:
            L_ = sum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum() for p in u["pieces"]); dens[id(u)] = L_ / max(1, len(t))
    med = float(np.median(list(dens.values()))) if dens else 1.0
    for t, insts in list(W.items()):
        W[t] = [u for u in insts if dens[id(u)] <= 1.7 * med]
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
    word_scores = []; chunk_scores = {}; chunk_worst_scores = {}
    for w, seg in zip(words, segs):
        if not seg: continue
        fr = word_frame(w["strokes"], seg)
        if not fr: continue
        baseline, xh = fr; sc = []
        for ch in seg:
            pieces = [w["strokes"][i][i0:i1].copy() for i, i0, i1 in ch["pieces"]]
            px = min(p[:, 0].min() for p in pieces); s_ = score(ch["char"], normalise(pieces, px, baseline, xh))
            if s_ is not None: sc.append(s_)
        word_scores.append((w["text"], (w["source"], w["page"], w["text"]), float(np.mean(sc)) if sc else None, float(np.max(sc)) if sc else None))
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
            chunk_worst_scores[((w["source"], w["page"], w["text"]), chars, run[0])] = float(np.max(scs)) if scs else None
    by_src = {src: sc for _, src, sc, _ in word_scores}; worst_by_src = {src: wst for _, src, _, wst in word_scores}
    vals = np.array([sc for _, _, sc, _ in word_scores if sc is not None]); cut = float(np.quantile(vals, 0.85))
    dropped = 0
    for t, insts in list(W.items()):
        kept = []
        for u in insts:
            u["score"] = by_src.get(u["src"]); u["worst"] = worst_by_src.get(u["src"])
            if u["score"] is not None and u["score"] > cut: dropped += 1
            elif u["worst"] is not None and u["worst"] > 1.1 * cut: dropped += 1   # one illegible letter spoils the word
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
    # strict quality tier: keep only the best 40% of letters by template score, and only chunks whose
    # WORST letter would pass that bar (mislabeled chunks hide a bad letter behind a good mean)
    # stroke-count sanity: the writer's stroke count per letter is consistent; fewer = incomplete (a one-line x),
    # many more = carrying a neighbour's fragment. Diacritics count, so use the mode over whole-stroke instances.
    import json, os
    for t, insts in list(L.items()):
        counts = collections.Counter(len(u["pieces"]) for u in insts if u.get("whole")) or collections.Counter(len(u["pieces"]) for u in insts)
        mode = counts.most_common(1)[0][0]
        if t.isupper(): L[t] = [u for u in insts if mode <= len(u["pieces"]) <= mode + 1]  # capitals vary (E, H: 2-4 strokes)
        else: L[t] = [u for u in insts if len(u["pieces"]) == mode]  # lowercase: exactly the writer's usual stroke count
    # curated blocklist (keyed by source: notebook, page, word, char index) survives rebuilds
    pre_block = {t: list(v) for t, v in L.items()}  # single-stroke halves of X live here before the blocklist removes them
    bl_path = "data/curation/blocklist.json"
    if os.path.exists(bl_path):
        block = {tuple(k) for k in json.load(open(bl_path))}
        for t, insts in list(L.items()): L[t] = [u for u in insts if tuple(u["src"]) + (u.get("k"),) not in block]
        for t, insts in list(W.items()):   # word-level entries carry None as the 4th field
            W[t] = [u for u in insts if tuple(u["src"]) + (None,) not in block]
            if not W[t]: del W[t]
    letter_scores = np.array([u["score"] for v in L.values() for u in v if u.get("score") is not None])
    strict = float(np.quantile(letter_scores, 0.40)) if len(letter_scores) else 1.0
    loose = {t: list(v) for t, v in L.items()}  # pool after stroke-count + blocklist, before the strict score cut
    for t, insts in list(L.items()):
        kept = [u for u in insts if u.get("score") is not None and u["score"] <= strict]
        # shape sanity for a few letters whose mis-cuts have a characteristic look
        if t == "k": kept = [u for u in kept if u["width"] >= 0.55 and len(u["pieces"]) >= 1 and u["height"] >= 1.6]
        if len(loose[t]) < 30: L[t] = loose[t]       # rare letter: keep the whole (stroke-count-filtered, unblocked) pool; curation decides
        elif len(kept) >= 4: L[t] = kept
        else: L[t] = sorted(loose[t], key=lambda u: u.get("score") if u.get("score") is not None else 9)[:max(4, len(loose[t]) // 3)]
    chunk_worst = {}
    for (src, chars, run0), sc in list(chunk_scores.items()): pass
    for t, insts in list(C.items()):
        kept = []
        for u in insts:
            worst = chunk_worst_scores.get((u["src"], t, u["run0"]))
            if worst is not None and worst <= strict: kept.append(u)
        if kept: C[t] = kept
        else: del C[t]
    # natural variants: every letter gets siblings bent/stretched/nudged from its real instances
    from .augment import augment_bank
    import random as _random
    # letters with no usable real seed are composed from the writer's own parts (j = i + g's descender)
    from .augment import compose_j, compose_x, compose_H, compose_D
    rj = _random.Random(3)
    for xc in "Xx":
        halves = [u for u in pre_block.get(xc, []) if len(u["pieces"]) == 1]
        if len([u for u in L.get(xc, []) if len(u["pieces"]) >= 2]) < 3 and len(halves) >= 2:
            xs = [x for x in (compose_x(halves, rj) for _ in range(12)) if x]
            if xs:
                for x in xs:
                    x["text"] = xc
                    if xc == "x":  # lowercase: squash to x-height
                        for p_ in x["pieces"]: p_[:, 1] *= 0.5; p_[:, :2] *= 1.0
                        x["height"] = float(np.ptp(np.concatenate([p_[:, 1] for p_ in x["pieces"]])))
                L[xc] = xs
    l_pool = [u for u in L.get("l", []) if u.get("whole") and len(u["pieces"]) == 1 and not u.get("synthetic")]
    eq_pool = [u for u in L.get("t", []) if not u.get("synthetic")]  # the t's crossbar donor list is replaced below if '=' is available
    try:
        import pickle as _pk; _S = _pk.load(open("data/derived/symbols.pkl", "rb")); eq_pool = [u for u in _S.get("=", []) if len(u["pieces"]) == 2] or eq_pool; paren_pool = [u for u in _S.get("(", []) if len(u["pieces"]) == 1]
    except Exception: paren_pool = []
    if len(L.get("H", [])) < 3 and l_pool and eq_pool: L["H"] = [compose_H(l_pool, eq_pool, rj) for _ in range(10)]
    if len(L.get("D", [])) < 3 and l_pool and paren_pool: L["D"] = [compose_D(l_pool, paren_pool, rj) for _ in range(10)]
    if len(L.get("j", [])) < 3:
        i_pool = [u for u in L.get("i", []) if u.get("whole") and len(u["pieces"]) == 2 and not u.get("synthetic")]
        g_pool = [u for u in L.get("g", []) if not u.get("synthetic")] + [u for u in L.get("y", []) if not u.get("synthetic")]
        if i_pool and g_pool:
            L["j"] = [compose_j(rj.choice(i_pool), rj.choice(g_pool), rj) for _ in range(12)]
    before = sum(map(len, L.values())); L = augment_bank(L, _random.Random(7)); report['_aug'] = (before, sum(map(len, L.values())))
    wb = sum(map(len, W.values())); W = augment_bank(W, _random.Random(5), per_instance=4, target_min=6, cap=40, strength=0.7); report['_waug'] = (wb, sum(map(len, W.values())))
    C = {}  # chunks are off: they cannot be curated letter by letter and mislabels slip through ("pp" that is not a pp)
    report['_strict'] = (strict, sum(map(len, L.values())), sum(map(len, C.values())))
    report['_words'] = tuple(wrep) + (dropped, cut)
    return dict(W), dict(C), dict(L), report


if __name__ == "__main__":
    d = pickle.load(open("data/derived/segments.pkl", "rb"))
    words = load_words(sorted({w["source"] for w in d["words"]}))
    W, C, L, rep = build(words, d["segs"])
    letter_h = {c: float(np.median([u["height"] for u in v])) for c, v in L.items() if len(v) >= 3}
    # expected vertical extent per letter (top, bottom) relative to the baseline, from clean whole-stroke instances
    letter_ext = {}
    for c, v in L.items():
        good = [u for u in v if u.get("whole")] or v
        if len(good) >= 3:
            tops = [min(p[:, 1].min() for p in u["pieces"]) for u in good]; bots = [max(p[:, 1].max() for p in u["pieces"]) for u in good]
            letter_ext[c] = (float(np.median(tops)), float(np.median(bots)))
    # geometric plausibility against the label: width ~ sum of typical letter widths, extent ~ union of letter extents
    wprior = d["prior"]; lw = {c: float(np.median([u["width"] for u in v])) for c, v in L.items() if len(v) >= 3}
    def plausible(u, tol_w=(0.65, 1.4)):
        t = u["text"]
        if not t.isalpha() or len(t) < 1: return True
        exp_w = sum(lw.get(c, wprior.get(c, 1.0)) for c in t) + 0.15 * (len(t) - 1)
        r = u["width"] / exp_w
        if not (tol_w[0] <= r <= tol_w[1]): return False
        ext = [letter_ext[c] for c in t if c in letter_ext]
        if ext:
            top = min(p[:, 1].min() for p in u["pieces"]); bot = max(p[:, 1].max() for p in u["pieces"])
            exp_top, exp_bot = min(e[0] for e in ext), max(e[1] for e in ext)
            if exp_top < -1.4 and top > exp_top + 0.7: return False     # should have an ascender/capital but doesn't
            if exp_top >= -1.4 and top < exp_top - 0.7: return False    # x-height-only text with a tall stroke in it
            if exp_bot > 0.5 and bot < exp_bot - 0.5: return False      # should descend but doesn't
            if exp_bot <= 0.5 and bot > exp_bot + 0.6: return False     # descends but shouldn't
        return True
    drop = {"chunks": 0, "letters": 0, "words": 0}
    for name, bank_, tol in (("chunks", C, (0.7, 1.35)), ("letters", L, (0.55, 1.6)), ("words", W, (0.65, 1.4))):
        for t, insts in list(bank_.items()):
            kept = [u for u in insts if plausible(u, tol)]; drop[name] += len(insts) - len(kept)
            if kept: bank_[t] = kept
            else: del bank_[t]
    print("implausible units dropped:", drop)
    caps = [letter_ext[c] for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c in letter_ext]
    if caps:
        cap_ext = (max(-2.2, float(np.median([e[0] for e in caps]))), float(np.median([e[1] for e in caps])))  # body-text capitals are ~2.2 xh, the sheet's are larger
        for dgt in "0123456789": letter_ext[dgt] = (-1.9, 0.1)  # digits: a little under capital height, same for all ten
        for cap in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            e = letter_ext.get(cap, cap_ext); letter_ext[cap] = (max(-2.3, e[0]), min(0.3, e[1]))
        for br in "()[]{}|": letter_ext[br] = (-2.1, 0.3)  # brackets: capital height, slight dip below the baseline
    pickle.dump({"words": W, "chunks": C, "letters": L, "letter_h": letter_h, "letter_ext": letter_ext}, open("data/derived/units.pkl", "wb"))
    print("letter extents (top,bottom in xh):", {c: (round(a, 2), round(b, 2)) for c, (a, b) in sorted(letter_ext.items()) if c in "aeghlpt"})
    print(f"words: {sum(map(len, W.values()))} instances / {len(W)} distinct | chunks: {sum(map(len, C.values()))} / {len(C)} | letters kept: {sum(map(len, L.values()))}")
    print("chunks dropped for template mismatch:", rep.pop("_chunks"))
    wa = rep.pop("_waug"); print(f"word variants: {wa[0]} -> {wa[1]}")
    ag = rep.pop("_aug"); print(f"augmentation: {ag[0]} real letter instances -> {ag[1]} with variants")
    st_ = rep.pop("_strict"); print(f"strict tier: letter score <= {st_[0]:.3f}; letters {st_[1]}, chunks {st_[2]}")
    wr = rep.pop("_words"); print(f"word filters: {wr[0]} -> {wr[1]} (outliers) -> {wr[1]-wr[2]} (template mismatch, cut {wr[3]:.2f})")
    nw = sum(1 for v in L.values() for i in v if i.get("whole")); print(f"letters that are complete strokes (no cut): {nw}/{sum(map(len, L.values()))}")
    print("capitals kept:", " ".join(f"{c}:{b}" for c, (a, b) in sorted(rep.items()) if c.isupper()))
