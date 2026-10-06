"""Harvest symbol and digit instances from lines whose text is known exactly.

For each (handwritten line, source line) pair: strip spaces from the source text,
map LaTeX-ish markup to plain characters (x_1 -> x1, 2^10 -> 210), and run the
template-guided DP over the whole line as if it were one word. Letters get shape
costs from the letter templates; symbols only get width costs. Then every
non-alphabetic character becomes a symbol instance, normalised by the line's
x-height (estimated from its lowercase letters).
"""
import pickle, re, collections
import numpy as np
from .align import load
from .recut import recut_word, build_templates
from .segment import default_width_prior
from .bank import XHEIGHT_LETTERS

WIDE_SYM = set("∞⇒→←↔∪∩Σ×"); NARROW_SYM = set("|[](){}.,;:!'1-")


def line_groups(strokes, gap_factor=0.6):
    """Cluster strokes into text lines by vertical position (centres), return lists of stroke ids top to bottom."""
    cy = np.array([s[:, 1].mean() for s in strokes]); h = np.array([s[:, 1].max() - s[:, 1].min() for s in strokes])
    order = np.argsort(cy); lines, cur, cur_y = [], [order[0]], cy[order[0]]
    typical = np.median(h[h > 2]) if (h > 2).any() else 10
    for i in order[1:]:
        if cy[i] - cur_y > gap_factor * typical * 1.6: lines.append(cur); cur = [i]
        else: cur.append(i)
        cur_y = np.mean(cy[cur])
    lines.append(cur)
    return [sorted(l, key=lambda i: strokes[i][:, 0].min()) for l in lines]


def flatten_source(line):
    t = line.replace("_", "").replace("^", "").replace(" ", "")
    return t


def harvest(page, source_lines, templates, prior):
    strokes = page["strokes"]; groups = line_groups(strokes)
    out = collections.defaultdict(list); frames = []
    # pair handwritten lines with source lines from the bottom up (the symbol block is last on the page)
    pairs = list(zip(groups[-len(source_lines):], source_lines))
    for ids, src in pairs:
        text = flatten_source(src); ls = [strokes[i] for i in ids]
        allp = np.vstack([s[:, :2] for s in ls]); h = allp[:, 1].max() - allp[:, 1].min()
        # rough frame: baseline at 75th percentile of y, x-height from line height / 2.4
        baseline = np.percentile(allp[:, 1], 75); xh = max(2.0, h / 2.6)
        pri = dict(prior)
        for c in text:
            if not c.isalnum(): pri.setdefault(c, 1.8 if c in WIDE_SYM else 0.5 if c in NARROW_SYM else 1.0)
        seg = recut_word(ls, text, pri, templates, baseline, xh, max_b=max(60, 3 * len(text)))
        if seg is None: continue
        # refine frame from lowercase letters, then normalise instances
        xs_pts = [np.vstack([ls[i][i0:i1, :2] for i, i0, i1 in ch["pieces"]]) for ch in seg]
        base_c = [np.percentile(p[:, 1], 92) for ch, p in zip(seg, xs_pts) if ch["char"] in XHEIGHT_LETTERS]
        top_c = [np.percentile(p[:, 1], 8) for ch, p in zip(seg, xs_pts) if ch["char"] in XHEIGHT_LETTERS]
        if len(base_c) >= 3: baseline = float(np.median(base_c)); xh = max(2.0, baseline - float(np.median(top_c)))
        frames.append((src, baseline, xh, len(seg)))
        for ch in seg:
            c = ch["char"]
            if c.isalpha(): continue
            pieces = [ls[i][i0:i1].copy() for i, i0, i1 in ch["pieces"]]
            px = min(p[:, 0].min() for p in pieces)
            norm = [np.column_stack([(p[:, 0] - px) / xh, (p[:, 1] - baseline) / xh, p[:, 2]]) for p in pieces]
            allq = np.vstack([p[:, :2] for p in norm])
            out[c].append({"text": c, "pieces": norm, "width": allq[:, 0].max(), "height": allq[:, 1].max() - allq[:, 1].min(),
                           "src": ("capitals_sheet", src), "joined_prev": ch["joined_prev"], "joined_next": ch["joined_next"]})
    return dict(out), frames


if __name__ == "__main__":
    pages = load("data/derived/capitals_sheet")
    src = [l for l in open("data/sources/capitals_sheet.txt").read().split("\n") if l.strip()]
    sym_lines = src[19:]  # the pseudocode + math block
    units = pickle.load(open("data/derived/units.pkl", "rb")); seg = pickle.load(open("data/derived/segments.pkl", "rb"))
    templates = build_templates(units["letters"])
    S, frames = harvest(pages[1], sym_lines, templates, seg["prior"])
    for f in frames: print(f"  line ok: {f[0][:40]:40s} xh={f[2]:.1f} chars={f[3]}")
    pickle.dump(S, open("data/derived/symbols.pkl", "wb"))
    print("symbol instances:", {k: len(v) for k, v in sorted(S.items())})
