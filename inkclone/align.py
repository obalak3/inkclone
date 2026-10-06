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


def word_ok(text, n_strokes):
    t = text.strip(".,;:!?()\"'")
    if not (t.isalpha() and len(t) >= 2 and t.lower() in DICT):
        return False
    return 0.3 * len(t) <= n_strokes <= 3 * len(t) + 2


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
    samples, unlabeled = [], []
    for wi, w in enumerate(page["words"]):
        idx = np.flatnonzero(owner == wi)
        if not len(idx): continue
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
