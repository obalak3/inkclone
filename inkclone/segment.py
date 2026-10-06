"""Cut labeled word samples into per-character stroke pieces.

A word is a sequence of strokes in pen order and a text label. Letters are written
left to right, so segmentation = choosing len(text)-1 cut points along the
concatenated sample sequence. Cuts at pen-ups are free; cuts inside a stroke (joined
letters) are allowed anywhere but cost a little, less near the baseline where the
connecting stroke runs. Each letter's x-extent is scored against a per-letter width
prior, which is re-estimated from the corpus (two EM-style passes).

Short strokes (i-dots, t-crosses, accents) are set aside first and attached afterwards
to the letter whose x-span they sit over.
"""
import json
import numpy as np

NARROW = set("iljtfr.,;:'!|()[]1"); WIDE = set("mwMW")


def default_width_prior():
    pri = {}
    for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
        pri[c] = 0.55 if c in NARROW else 1.6 if c in WIDE else 1.0
    return pri


def split_diacritics(strokes, word_h):
    """-> (main strokes with their original index, small strokes)."""
    main, small = [], []
    for i, s in enumerate(strokes):
        ext = s[:, :2].max(0) - s[:, :2].min(0)
        if max(ext) < 0.3 * word_h or (ext[1] < 0.12 * word_h and ext[0] < 0.6 * word_h):
            small.append((i, s))
        else:
            main.append((i, s))
    return main, small


def segment_word(strokes, text, prior, unit=None, lam_join=0.6, lam_width=1.0):
    """-> list of per-character dicts: {char, pieces: [(stroke_idx, i0, i1)], joined_prev, joined_next}
    or None if the word cannot be segmented (too few samples, mismatch)."""
    K = len(text)
    allpts = np.vstack([s[:, :2] for s in strokes])
    word_h = allpts[:, 1].max() - allpts[:, 1].min()
    word_w = allpts[:, 0].max() - allpts[:, 0].min()
    main, small = split_diacritics(strokes, word_h)
    if not main:
        return None
    if K == 1:
        return [{"char": text, "pieces": [(i, 0, len(s)) for i, s in main] + [(i, 0, len(s)) for i, s in small],
                 "joined_prev": False, "joined_next": False}]
    # concatenated sample sequence with (stroke_idx, local_idx) bookkeeping
    seq, owner = [], []
    for i, s in main:
        seq.append(s[:, :2]); owner.append(np.column_stack([np.full(len(s), i), np.arange(len(s))]))
    seq = np.vstack(seq); owner = np.vstack(owner); N = len(seq)
    starts = np.cumsum([0] + [len(s) for _, s in main])  # stroke start indices in seq
    penup = set(starts[1:-1].tolist())
    # candidate cuts: every ~0.4pt of travel, plus all pen-ups
    step = max(1, int(N / max(60, 4 * K * 6)))
    cands = sorted(set(range(step, N - step, step)) | penup)
    cands = [c for c in cands if 0 < c < N]
    if len(cands) < K - 1:
        return None
    # expected width per char in pt: prior units scaled so they sum to the word width
    pw = np.array([prior.get(c, 1.0) for c in text]); unit = unit or word_w / pw.sum()
    exp_w = pw * unit
    ybot = np.percentile(seq[:, 1], 85)  # baseline-ish (y grows downward)
    def join_cost(c):
        if c in penup: return 0.0
        return lam_join * (1.0 + max(0.0, (ybot - seq[c, 1]) / word_h))  # cheaper near baseline
    def width_cost(a, b, k):
        seg = seq[a:b]; w = seg[:, 0].max() - seg[:, 0].min() if len(seg) else 0.0
        return lam_width * ((w - exp_w[k]) / (0.5 * exp_w[k] + 0.3 * unit)) ** 2
    # DP over (char k, cut index j): best[k][j] = cost of chars 0..k ending at cands[j]
    C = len(cands); INF = 1e18
    best = np.full((K, C), INF); back = np.full((K, C), -1, int)
    for j, c in enumerate(cands):
        best[0, j] = width_cost(0, c, 0) + join_cost(c)
    for k in range(1, K - 1):
        for j, c in enumerate(cands):
            for jp in range(j):
                v = best[k-1, jp] + width_cost(cands[jp], c, k) + join_cost(c)
                if v < best[k, j]: best[k, j] = v; back[k, j] = jp
    # last char runs from cut to end
    finals = [best[K-2, j] + width_cost(cands[j], N, K-1) for j in range(C)]
    j = int(np.argmin(finals)); cuts = [cands[j]]
    for k in range(K - 2, 0, -1):
        j = back[k, j]; cuts.append(cands[j])
    cuts = cuts[::-1]; bounds = [0] + cuts + [N]
    chars = []
    for k in range(K):
        a, b = bounds[k], bounds[k+1]
        pieces = []
        for i, s in main:
            s0 = starts[[m for m, (ii, _) in enumerate(main) if ii == i][0]]
            lo, hi = max(a, s0), min(b, s0 + len(s))
            if hi > lo: pieces.append((i, lo - s0, hi - s0))
        chars.append({"char": text[k], "pieces": pieces,
                      "joined_prev": k > 0 and bounds[k] not in penup,
                      "joined_next": k < K-1 and bounds[k+1] not in penup})
    # attach diacritics to the char whose x-span is nearest their centre
    spans = []
    for ch in chars:
        xs = np.concatenate([strokes[i][i0:i1, 0] for i, i0, i1 in ch["pieces"]]) if ch["pieces"] else np.array([np.nan])
        spans.append((np.nanmin(xs), np.nanmax(xs)))
    for i, s in small:
        cx = s[:, 0].mean()
        d = [0 if lo <= cx <= hi else min(abs(cx - lo), abs(cx - hi)) for lo, hi in spans]
        chars[int(np.argmin(d))]["pieces"].append((i, 0, len(s)))
    return chars


def char_width(strokes, ch):
    xs = np.concatenate([strokes[i][i0:i1, 0] for i, i0, i1 in ch["pieces"]])
    return xs.max() - xs.min()


def segment_corpus(words, passes=2):
    """words: list of {text, strokes}. Returns list of segmentations (None where failed)
    and the learned width prior (per char, relative units)."""
    prior = default_width_prior()
    segs = None
    for _ in range(passes):
        segs = [segment_word(w["strokes"], w["text"], prior) for w in words]
        acc = {}
        for w, sg in zip(words, segs):
            if not sg: continue
            ws = np.array([char_width(w["strokes"], ch) for ch in sg]); mean = ws.mean() or 1.0
            for ch, cw in zip(sg, ws): acc.setdefault(ch["char"], []).append(cw / mean)
        for c, v in acc.items():
            if len(v) >= 3: prior[c] = float(np.median(v))
    return segs, prior


def load_words(prefixes):
    words = []
    for prefix in prefixes:
        arrs = np.load(prefix + ".npz"); meta = json.load(open(prefix + ".json"))
        for k, m in enumerate(meta):
            words.append({**m, "strokes": [arrs[f"w{k}_s{j}"] for j in range(m["n_strokes"])], "source": prefix})
    return words


if __name__ == "__main__":
    import sys, glob, collections, pickle
    prefixes = [p[:-4] for p in sorted(glob.glob("data/derived/*.words.npz"))]
    words = load_words(prefixes)
    segs, prior = segment_corpus(words)
    ok = sum(1 for s in segs if s)
    n_chars = sum(len(s) for s in segs if s); joins = sum(ch["joined_next"] for s in segs if s for ch in s)
    print(f"{len(words)} words, {ok} segmented, {n_chars} character instances, {joins} joined letter pairs ({joins/max(1,n_chars-ok):.0%} of adjacent pairs)")
    cnt = collections.Counter(ch["char"] for s in segs if s for ch in s)
    print("instances per letter:", " ".join(f"{c}:{n}" for c, n in sorted(cnt.items())))
    print("learned width prior (rel):", {c: round(v, 2) for c, v in sorted(prior.items()) if c.islower()})
    pickle.dump({"words": [{k: v for k, v in w.items() if k != "strokes"} for w in words], "segs": segs, "prior": prior}, open("data/derived/segments.pkl", "wb"))
