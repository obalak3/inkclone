"""Pair GoodNotes OCR word boxes with the strokes inside them -> labeled word samples.

Keeps only words that look trustworthy: alphabetic, in the system dictionary (or a
short common-word list), with a plausible stroke count for their length. Equations
and diagrams fall through as unlabeled strokes, which is intended.
"""
import json, re
import numpy as np

COMMON = set("a i is of the and or to in at if as it an by on be we so no do up".split())
try:
    DICT = {w.strip().lower() for w in open("/usr/share/dict/words")} | COMMON
except FileNotFoundError:
    DICT = COMMON


def load(prefix):
    arrs = np.load(prefix + ".npz"); meta = json.load(open(prefix + ".json"))
    pages = []
    for pi, m in enumerate(meta):
        m = dict(m); m["strokes"] = [arrs[f"p{pi}_s{si}"] for si in range(m["n_strokes"])]
        pages.append(m)
    return pages


def is_word(text):
    t = text.strip(".,;:!?()\"'")
    return t.isalpha() and t.lower() in DICT and (len(t) >= 3 or t.lower() in COMMON)


def word_ok(text, n_strokes):
    t = text.strip(".,;:!?()\"'")
    return is_word(t) and 0.3 * len(t) <= n_strokes <= 2 * len(t) + 1


def line_trust(words, min_tokens=2, min_rate=0.6):
    """Group OCR boxes into text lines by vertical overlap. A line is trusted when most
    of its tokens are dictionary words (equation lines OCR into garbage tokens).
    Returns a bool per word."""
    trusted = [False] * len(words)
    order = sorted(range(len(words)), key=lambda i: (words[i]["bbox"][1], words[i]["bbox"][0]))
    lines = []
    for i in order:
        y0, y1 = words[i]["bbox"][1], words[i]["bbox"][3]
        for ln in lines:
            ly0, ly1 = ln["y"]
            if min(y1, ly1) - max(y0, ly0) > 0.5 * min(y1 - y0, ly1 - ly0):
                ln["ids"].append(i); ln["y"] = (min(y0, ly0), max(y1, ly1)); break
        else:
            lines.append({"ids": [i], "y": (y0, y1)})
    for ln in lines:
        ids = ln["ids"]
        rate = sum(is_word(words[i]["text"]) for i in ids) / len(ids)
        if len(ids) >= min_tokens and rate >= min_rate:
            for i in ids: trusted[i] = True
    return trusted


def assign_strokes(strokes, words, pad=1.0):
    """Each stroke goes to the word box containing the largest share of its samples."""
    boxes = np.array([w["bbox"] for w in words]) if words else np.zeros((0, 4))
    owner = np.full(len(strokes), -1)
    for si, s in enumerate(strokes):
        if not len(boxes): break
        x, y = s[:, 0][:, None], s[:, 1][:, None]
        inside = (x >= boxes[:, 0]-pad) & (x <= boxes[:, 2]+pad) & (y >= boxes[:, 1]-pad) & (y <= boxes[:, 3]+pad)
        share = inside.mean(0)
        if share.max() >= 0.6:
            owner[si] = share.argmax()
    return owner


def align_page(page, page_index):
    owner = assign_strokes(page["strokes"], page["words"])
    trusted = line_trust(page["words"])
    samples, unlabeled = [], []
    for wi, w in enumerate(page["words"]):
        idx = np.flatnonzero(owner == wi)
        if not len(idx) or not trusted[wi]: continue
        strokes = [page["strokes"][i] for i in idx]
        if word_ok(w["text"], len(strokes)):
            samples.append({"text": w["text"].strip(".,;:!?()\"'"), "page": page_index, "bbox": w["bbox"],
                            "stroke_ids": idx.tolist(), "strokes": strokes})
    unlabeled = [page["strokes"][i] for i in np.flatnonzero(owner == -1)]
    return samples, unlabeled


def align(pages):
    all_samples, n_unlabeled = [], 0
    for pi, pg in enumerate(pages):
        s, u = align_page(pg, pi); all_samples += s; n_unlabeled += len(u)
    return all_samples, n_unlabeled


def save_samples(samples, prefix):
    arrs, meta = {}, []
    for k, s in enumerate(samples):
        for j, st in enumerate(s["strokes"]): arrs[f"w{k}_s{j}"] = st.astype(np.float32)
        meta.append({kk: v for kk, v in s.items() if kk != "strokes"} | {"n_strokes": len(s["strokes"])})
    np.savez_compressed(prefix + ".npz", **arrs); json.dump(meta, open(prefix + ".json", "w"))


if __name__ == "__main__":
    import sys, collections
    prefix = sys.argv[1]
    pages = load(prefix); samples, n_un = align(pages)
    out = prefix + ".words"; save_samples(samples, out)
    n_strokes = sum(len(p["strokes"]) for p in pages); n_lab = sum(len(s["strokes"]) for s in samples)
    print(f"{len(samples)} labeled words ({len(set(s['text'].lower() for s in samples))} distinct), "
          f"{n_lab}/{n_strokes} strokes labeled ({n_lab/n_strokes:.0%}), {n_un} strokes in no word box -> {out}")
    print(collections.Counter(s["text"].lower() for s in samples).most_common(20))
    chars = collections.Counter(c for s in samples for c in s["text"])
    print("chars covered:", "".join(sorted(chars)), "| missing lowercase:", "".join(c for c in "abcdefghijklmnopqrstuvwxyz" if c not in chars))
