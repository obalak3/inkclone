"""Build the glyph bank + style statistics from segmented words.

Every character instance is stored in word-normalised coordinates: baseline at y=0,
one x-height = 1 unit, x measured from the instance's own left edge. Alongside each
instance we keep whether it was joined to its neighbours and where the pen entered
and left it, so the synthesiser can chain joined letters into one stroke.

Style statistics measured from the same data: join rate per letter pair, gap between
unjoined neighbours, and the per-word slant.
"""
import pickle, json, collections
import numpy as np
from .segment import load_words

XHEIGHT_LETTERS = set("acemnorsuvwxz"); DESCENDERS = set("gjpqy"); ASCENDERS = set("bdfhklt")
# plausible normalised ink height per letter class (in x-heights): rejects bad frames/cuts
HEIGHT_RANGE = {"x": (0.55, 1.8), "asc": (1.1, 3.2), "desc": (1.1, 3.2), "cap": (1.8, 5.5), "other": (0.3, 5.5)}


def letter_class(c):
    if c in XHEIGHT_LETTERS: return "x"
    if c in ASCENDERS: return "asc"
    if c in DESCENDERS: return "desc"
    if c.isupper(): return "cap"
    return "other"


def word_frame(strokes, seg):
    """-> (baseline_y, xheight) for a segmented word, or None."""
    base_pts, top_pts = [], []
    for ch in seg:
        pts = np.vstack([strokes[i][i0:i1, :2] for i, i0, i1 in ch["pieces"]])
        if ch["char"] not in DESCENDERS: base_pts.append(np.percentile(pts[:, 1], 92))
        if ch["char"] in XHEIGHT_LETTERS: top_pts.append(np.percentile(pts[:, 1], 8))
    if not base_pts: return None
    baseline = float(np.median(base_pts))
    if top_pts:
        xh = baseline - float(np.median(top_pts))
    else:
        allp = np.vstack([strokes[i][i0:i1, :2] for ch in seg for i, i0, i1 in ch["pieces"]])
        xh = 0.5 * (allp[:, 1].max() - allp[:, 1].min())
    return (baseline, xh) if xh > 0.5 else None


def slant_deg(pts):
    """Slant of near-vertical travel: angle of the principal axis of downstroke vectors."""
    d = np.diff(pts, axis=0); d = d[np.abs(d[:, 1]) > 2 * np.abs(d[:, 0])]
    if len(d) < 5: return None
    d = d * np.sign(d[:, 1:2]); return float(np.degrees(np.arctan2(d[:, 0].sum(), d[:, 1].sum())))


def build(words, segs):
    bank = collections.defaultdict(list); joins = collections.Counter(); pairs = collections.Counter()
    gaps = collections.defaultdict(list); slants = []; word_xh = []
    for w, seg in zip(words, segs):
        if not seg: continue
        fr = word_frame(w["strokes"], seg)
        if not fr: continue
        baseline, xh = fr; word_xh.append(xh)
        allp = np.vstack([s[:, :2] for s in w["strokes"]]); sl = slant_deg(allp)
        if sl is not None: slants.append(sl)
        n_main = len([1 for st in w["strokes"]])  # pieces listed main-first, diacritics appended
        placed = []
        for k, ch in enumerate(seg):
            pieces = [w["strokes"][i][i0:i1].copy() for i, i0, i1 in ch["pieces"]]
            # main pieces = those cut from the big strokes (first entries); diacritics were appended after
            n_dia = sum(1 for i, i0, i1 in ch["pieces"] if i0 == 0 and i1 == len(w["strokes"][i]) and
                        max(w["strokes"][i][:, :2].max(0) - w["strokes"][i][:, :2].min(0)) < 0.3 * (allp[:, 1].max() - allp[:, 1].min()))
            main = pieces[:len(pieces) - n_dia] or pieces
            mx0, mx1 = min(p[:, 0].min() for p in main), max(p[:, 0].max() for p in main)
            ys = np.concatenate([p[:, 1] for p in pieces]); h = (ys.max() - ys.min()) / xh
            lo, hi = HEIGHT_RANGE[letter_class(ch["char"])]
            ok = lo <= h <= hi and 0.15 <= (mx1 - mx0) / xh <= 3.5
            # pen width stays in pt (a pen does not get thicker when you write bigger)
            norm = [np.column_stack([(p[:, 0] - mx0) / xh, (p[:, 1] - baseline) / xh, p[:, 2]]) for p in pieces]
            inst = {"char": ch["char"], "pieces": norm, "width": (mx1 - mx0) / xh, "ok": ok,
                    "joined_prev": ch["joined_prev"], "joined_next": ch["joined_next"],
                    "entry": norm[0][0, :2].tolist(), "exit": norm[0][-1, :2].tolist(),
                    "x0": mx0, "x1": mx1, "advance": None,
                    "source": (w["source"], w["page"], w["text"], k)}
            placed.append(inst)
        for k, inst in enumerate(placed):
            if k + 1 < len(placed):
                nxt = placed[k + 1]; adv = (nxt["x0"] - inst["x0"]) / xh
                if inst["joined_next"]: inst["advance"] = adv
                pair = inst["char"] + nxt["char"]; pairs[pair] += 1
                if inst["joined_next"]: joins[pair] += 1
                else: gaps[pair].append((nxt["x0"] - inst["x1"]) / xh)
            if inst["ok"]:
                inst.pop("x0"); inst.pop("x1"); bank[inst["char"]].append(inst)
    all_gaps = np.array([g for v in gaps.values() for g in v])
    pen = float(np.median(np.concatenate([s[:, 2] for w in words for s in w["strokes"]])))
    stats = {"pen_width_pt": pen, "join_rate": {p: joins[p] / pairs[p] for p in pairs}, "pair_count": dict(pairs),
             "global_join_rate": sum(joins.values()) / max(1, sum(pairs.values())),
             "gap_median": float(np.median(all_gaps)), "gap_iqr": [float(np.percentile(all_gaps, 25)), float(np.percentile(all_gaps, 75))],
             "gap_by_pair": {p: float(np.median(v)) for p, v in gaps.items() if len(v) >= 3},
             "slant_deg_median": float(np.median(slants)), "slant_deg_iqr": [float(np.percentile(slants, 25)), float(np.percentile(slants, 75))],
             "xheight_pt_median": float(np.median(word_xh)), "xheight_pt_iqr": [float(np.percentile(word_xh, 25)), float(np.percentile(word_xh, 75))]}
    return dict(bank), stats


if __name__ == "__main__":
    d = pickle.load(open("data/derived/segments.pkl", "rb"))
    words = load_words(sorted({w["source"] for w in d["words"]}))
    bank, stats = build(words, d["segs"])
    pickle.dump({"bank": bank, "stats": stats}, open("data/derived/bank.pkl", "wb"))
    n = sum(len(v) for v in bank.values())
    print(f"bank: {n} instances over {len(bank)} characters")
    print(f"x-height {stats['xheight_pt_median']:.1f}pt (IQR {stats['xheight_pt_iqr'][0]:.1f}..{stats['xheight_pt_iqr'][1]:.1f}); "
          f"slant {stats['slant_deg_median']:.1f} deg (IQR {stats['slant_deg_iqr'][0]:.1f}..{stats['slant_deg_iqr'][1]:.1f}); "
          f"unjoined letter gap {stats['gap_median']:.2f} xh (IQR {stats['gap_iqr'][0]:.2f}..{stats['gap_iqr'][1]:.2f}); "
          f"global join rate {stats['global_join_rate']:.0%}")
    jr = stats["join_rate"]; pc = stats["pair_count"]
    common = sorted((p for p in pc if pc[p] >= 12), key=lambda p: -jr[p])
    print("most-joined pairs:", ", ".join(f"{p} {jr[p]:.0%} (n={pc[p]})" for p in common[:12]))
    print("least-joined pairs:", ", ".join(f"{p} {jr[p]:.0%} (n={pc[p]})" for p in common[-10:]))
    # which letters does he join OUT of vs INTO
    out_of = collections.defaultdict(lambda: [0, 0]); into = collections.defaultdict(lambda: [0, 0])
    for p, n_ in pc.items():
        out_of[p[0]][0] += jr[p] * n_; out_of[p[0]][1] += n_; into[p[1]][0] += jr[p] * n_; into[p[1]][1] += n_
    print("join rate OUT OF letter:", " ".join(f"{c}:{a/b:.0%}" for c, (a, b) in sorted(out_of.items()) if b >= 15 and c.islower()))
    print("join rate INTO letter:  ", " ".join(f"{c}:{a/b:.0%}" for c, (a, b) in sorted(into.items()) if b >= 15 and c.islower()))
