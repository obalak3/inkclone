"""Label a handwritten page from its typed source text instead of the OCR text.

GoodNotes' OCR boxes are good at *where* the words are and bad at *what* they say.
When we know what the writer copied, align the OCR boxes (in reading order) to the
source tokens with a fuzzy sequence alignment (1 or 2 boxes per token, skips on both
sides), and label each matched box group with the source token.
"""
import re, json, difflib
import numpy as np
from .align import load, assign_strokes, save_samples

PUNCT = ".,;:!?()\"'"


def reading_order(words):
    """Group OCR boxes into lines by vertical overlap, return boxes sorted line by line."""
    lines = []
    for w in sorted(words, key=lambda w: (w["bbox"][1], w["bbox"][0])):
        y0, y1 = w["bbox"][1], w["bbox"][3]
        for ln in lines:
            ly0, ly1 = ln["y"]
            if min(y1, ly1) - max(y0, ly0) > 0.4 * min(y1 - y0, ly1 - ly0):
                ln["ws"].append(w); ln["y"] = (min(y0, ly0), max(y1, ly1)); break
        else:
            lines.append({"ws": [w], "y": (y0, y1)})
    lines.sort(key=lambda ln: ln["y"][0])
    out = []
    for ln in lines: out += sorted(ln["ws"], key=lambda w: w["bbox"][0])
    return out


def sim(a, b):
    a, b = a.strip(PUNCT).lower(), b.strip(PUNCT).lower()
    if not a or not b: return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def align_tokens(boxes, tokens, gap=-0.45, min_sim=0.5):
    """Needleman-Wunsch over OCR boxes (i) and source tokens (j). Moves: 1 box->1 token,
    2 boxes->1 token (merged text), skip box, skip token. Returns list of (box_ids, token_idx)."""
    n, m = len(boxes), len(tokens)
    S = np.full((n + 1, m + 1), -1e9); B = {}
    S[0, :] = np.arange(m + 1) * gap; S[:, 0] = np.arange(n + 1) * gap
    txt = [b["text"] for b in boxes]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best, mv = S[i-1, j] + gap, ("skipb",)
            if S[i, j-1] + gap > best: best, mv = S[i, j-1] + gap, ("skipt",)
            s1 = sim(txt[i-1], tokens[j-1])
            if s1 >= min_sim and S[i-1, j-1] + s1 > best: best, mv = S[i-1, j-1] + s1, ("m1",)
            if i >= 2:
                s2 = sim(txt[i-2] + txt[i-1], tokens[j-1])
                if s2 >= min_sim and S[i-2, j-1] + s2 - 0.1 > best: best, mv = S[i-2, j-1] + s2 - 0.1, ("m2",)
            S[i, j] = best; B[i, j] = mv
    i, j, out = n, m, []
    while i > 0 and j > 0:
        mv = B[i, j][0]
        if mv == "m1": out.append(([i-1], j-1)); i -= 1; j -= 1
        elif mv == "m2": out.append(([i-2, i-1], j-1)); i -= 2; j -= 1
        elif mv == "skipb": i -= 1
        else: j -= 1
    return out[::-1]


def trim_edge_punct(strokes, label):
    """Drop small strokes (comma/period sized) hanging off the left edge of the word, and off
    the right edge when the label carries no trailing punctuation."""
    if len(strokes) < 2: return strokes
    allp = np.vstack([s[:, :2] for s in strokes]); h = allp[:, 1].max() - allp[:, 1].min()
    ext = [s[:, :2].max(0) - s[:, :2].min(0) for s in strokes]
    small = [max(e) < 0.3 * h for e in ext]
    big_x0 = min(s[:, 0].min() for s, sm in zip(strokes, small) if not sm) if not all(small) else allp[:, 0].min()
    big_x1 = max(s[:, 0].max() for s, sm in zip(strokes, small) if not sm) if not all(small) else allp[:, 0].max()
    keep = []
    for s, sm in zip(strokes, small):
        cx = s[:, 0].mean()
        if sm and cx < big_x0 - 0.05 * h: continue
        if sm and label[-1] not in PUNCT and cx > big_x1 + 0.05 * h and s[:, 1].mean() > allp[:, 1].min() + 0.6 * h: continue
        keep.append(s)
    return keep


def label_pages(pages, source_text):
    tokens = [t for t in re.findall(r"\S+", source_text) if re.search(r"[A-Za-z]", t)]
    boxes, page_of = [], []
    for pi, pg in enumerate(pages):
        ws = reading_order(pg["words"]); boxes += ws; page_of += [pi] * len(ws)
    pairs = align_tokens(boxes, tokens)
    samples, used_tok = [], set()
    per_page_boxes = {}
    for ids, tj in pairs:
        tok = tokens[tj]; ocr = "".join(boxes[i]["text"] for i in ids)
        # keep trailing punctuation in the label only if the OCR box also captured it
        label = tok if (tok[-1] in PUNCT and ocr[-1:] in PUNCT) else tok.rstrip(PUNCT)
        label = label.lstrip(PUNCT)
        if not re.search(r"[A-Za-z]", label): continue
        bb = np.array([boxes[i]["bbox"] for i in ids]); bbox = [bb[:, 0].min(), bb[:, 1].min(), bb[:, 2].max(), bb[:, 3].max()]
        per_page_boxes.setdefault(page_of[ids[0]], []).append({"bbox": bbox, "text": label}); used_tok.add(tj)
    for pi, wl in per_page_boxes.items():
        owner = assign_strokes(pages[pi]["strokes"], wl)
        for wi, w in enumerate(wl):
            idx = np.flatnonzero(owner == wi)
            if len(idx) < 1: continue
            n_letters = len(w["text"].strip(PUNCT))
            if not (0.3 * n_letters <= len(idx) <= 2.5 * n_letters + 2): continue
            strokes = trim_edge_punct([pages[pi]["strokes"][i] for i in idx], w["text"])
            samples.append({"text": w["text"], "page": pi, "bbox": [round(v, 2) for v in w["bbox"]],
                            "stroke_ids": idx.tolist(), "strokes": strokes})
    return samples, len(tokens), len(used_tok)


if __name__ == "__main__":
    import sys
    prefix, src = sys.argv[1], sys.argv[2]
    pages = load(prefix); samples, n_tok, n_used = label_pages(pages, open(src).read())
    save_samples(samples, prefix + ".words")
    print(f"{n_used}/{n_tok} source tokens matched to OCR boxes, {len(samples)} labeled words saved -> {prefix}.words")
    caps = sum(1 for s in samples if s["text"][0].isupper()); print(f"  {caps} start with a capital; sample: {[s['text'] for s in samples[:12]]}")
