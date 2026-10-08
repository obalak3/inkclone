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


def pair_lines(groups, strokes, ocr_words, source_lines, min_sim=0.3):
    """Monotonic alignment of handwritten line groups to source lines using the OCR letters in each band."""
    import difflib
    def band_text(ids):
        ys = np.concatenate([strokes[i][:, 1] for i in ids]); y0, y1 = ys.min(), ys.max()
        ws = [w["text"] for w in ocr_words if y0 <= (w["bbox"][1] + w["bbox"][3]) / 2 <= y1]
        return re.sub(r"[^a-z0-9]", "", " ".join(ws).lower())
    gt = [band_text(g) for g in groups]; st = [re.sub(r"[^a-z0-9]", "", l.replace("_", "").replace("^", "").lower()) for l in source_lines]
    n, m = len(groups), len(source_lines); gap = -0.2
    S = np.full((n + 1, m + 1), -1e9); S[0, :] = np.arange(m + 1) * gap; S[:, 0] = np.arange(n + 1) * gap; B = {}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sim = difflib.SequenceMatcher(None, gt[i-1], st[j-1]).ratio() if gt[i-1] and st[j-1] else 0.0
            opts = [(S[i-1, j] + gap, "g"), (S[i, j-1] + gap, "s")]
            if sim >= min_sim: opts.append((S[i-1, j-1] + sim, "m"))
            S[i, j], B[i, j] = max(opts)
    i, j, pairs = n, m, []
    while i > 0 and j > 0:
        mv = B[i, j]
        if mv == "m": pairs.append((groups[i-1], source_lines[j-1])); i -= 1; j -= 1
        elif mv == "g": i -= 1
        else: j -= 1
    return pairs[::-1]


def harvest(page, source_lines, templates, prior):
    strokes = page["strokes"]; groups = line_groups(strokes)
    out = collections.defaultdict(list); frames = []
    pairs = pair_lines(groups, strokes, page["words"], source_lines)
    for ids, src in pairs:
        ls = [strokes[i] for i in ids]
        allp = np.vstack([s_[:, :2] for s_ in ls]); h = allp[:, 1].max() - allp[:, 1].min()
        baseline = np.percentile(allp[:, 1], 75); xh = max(2.0, h / 2.6)
        # ink clusters along x: strokes whose x-ranges are within a small gap of each other
        order = sorted(range(len(ls)), key=lambda k: ls[k][:, 0].min()); clusters, cur, cur_x1 = [], [], None
        for k in order:
            x0, x1 = ls[k][:, 0].min(), ls[k][:, 0].max()
            if cur and x0 - cur_x1 > 0.45 * xh: clusters.append(cur); cur = []
            cur.append(k); cur_x1 = max(cur_x1, x1) if cur_x1 is not None and cur else x1
            cur_x1 = max(cur_x1, x1)
        clusters.append(cur)
        tokens = [flatten_source(t) for t in src.split()]
        # align clusters to tokens by sequence: cost = relative width mismatch, with skips
        pri = dict(prior)
        for c in "".join(tokens):
            if not c.isalnum(): pri.setdefault(c, 1.8 if c in WIDE_SYM else 0.5 if c in NARROW_SYM else 1.0)
        tw = np.array([sum(pri.get(c, 1.0) for c in t) for t in tokens]); cw = np.array([max(ls[k][:, 0].max() for k in cl) - min(ls[k][:, 0].min() for k in cl) for cl in clusters])
        unit = cw.sum() / tw.sum()
        n, m = len(clusters), len(tokens); S = np.full((n + 1, m + 1), 1e9); S[0, 0] = 0; Bk = {}
        for i in range(0, n + 1):
            for j in range(0, m + 1):
                if i == 0 and j == 0: continue
                opts = []
                if i > 0: opts.append((S[i-1, j] + 1.5, "g"))
                if j > 0: opts.append((S[i, j-1] + 1.5, "s"))
                if i > 0 and j > 0: opts.append((S[i-1, j-1] + abs(cw[i-1] - tw[j-1] * unit) / (unit * max(tw[j-1], 0.5)), "m"))
                if i > 1 and j > 0:  # two clusters -> one token (a token with an internal gap)
                    w2 = max(ls[k][:, 0].max() for k in clusters[i-1] + clusters[i-2]) - min(ls[k][:, 0].min() for k in clusters[i-1] + clusters[i-2])
                    opts.append((S[i-2, j-1] + 0.3 + abs(w2 - tw[j-1] * unit) / (unit * max(tw[j-1], 0.5)), "m2"))
                S[i, j], Bk[i, j] = min(opts)
        i, j, matched = n, m, []
        while i > 0 or j > 0:
            mv = Bk.get((i, j), "g" if i > 0 else "s")
            if mv == "m": matched.append((clusters[i-1], tokens[j-1])); i -= 1; j -= 1
            elif mv == "m2": matched.append((clusters[i-2] + clusters[i-1], tokens[j-1])); i -= 2; j -= 1
            elif mv == "g": i -= 1
            else: j -= 1
        matched = matched[::-1]
        # x-height from lowercase letters among single-letter tokens, if any
        frames.append((src, baseline, xh, len(matched), m))
        for cl, tok in matched:
            tls = [ls[k] for k in cl]
            if len(tok) == 1:
                seg = [{"char": tok, "pieces": [(k, 0, len(tls[k])) for k in range(len(tls))], "joined_prev": False, "joined_next": False}]
            else:
                seg = recut_word(tls, tok, pri, templates, baseline, xh, max_b=max(40, 6 * len(tok)))
                if seg is None: continue
            occ = {}
            for ch in seg:
                c = ch["char"]
                if c.isalpha(): continue
                occ[c] = occ.get(c, 0) + 1
                pieces = [tls[i][i0:i1].copy() for i, i0, i1 in ch["pieces"]]
                px = min(p_[:, 0].min() for p_ in pieces)
                norm = [np.column_stack([(p_[:, 0] - px) / xh, (p_[:, 1] - baseline) / xh, p_[:, 2]]) for p_ in pieces]
                allq = np.vstack([p_[:, :2] for p_ in norm])
                out[c].append({"text": c, "pieces": norm, "width": allq[:, 0].max(), "height": allq[:, 1].max() - allq[:, 1].min(),
                               "src": ("capitals_sheet", src, tok, occ[c]), "single": len(tok) == 1,
                               "joined_prev": ch["joined_prev"], "joined_next": ch["joined_next"]})
    return dict(out), frames


if __name__ == "__main__":
    pages = load("data/derived/capitals_sheet")
    src = [l for l in open("data/sources/capitals_sheet.txt").read().split("\n") if l.strip()]
    sym_lines = src[19:]  # the pseudocode + math block
    units = pickle.load(open("data/derived/units.pkl", "rb")); seg = pickle.load(open("data/derived/segments.pkl", "rb"))
    templates = build_templates(units["letters"])
    S, frames = harvest(pages[1], sym_lines, templates, seg["prior"])
    for f in frames: print(f"  line: {f[0][:44]:44s} tokens matched {f[3]}/{f[4]}")
    pickle.dump(S, open("data/derived/symbols.pkl", "wb"))
    print("symbol instances:", {k: len(v) for k, v in sorted(S.items())})
