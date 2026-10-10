"""Digit instances from the notebooks: GoodNotes' OCR layer puts a box around lone digits.
Take the strokes inside each single-digit box (1-2 strokes), normalise to digit height,
and keep the ones whose shape matches the hand-judged sheet seed for that digit."""
import glob, json, pickle
import numpy as np
from .align import load, assign_strokes
from .units import resample

DIGIT_TOP, DIGIT_BOT = -1.9, 0.1


def shape(pieces):
    main = max(pieces, key=len)[:, :2]; r = resample(main - main.min(0), 20)
    return np.r_[r.ravel() / max(np.ptp(main[:, 1]), 1e-6), len(pieces)]


def normalise(strokes):
    allp = np.vstack([s[:, :2] for s in strokes]); top, bot = allp[:, 1].min(), allp[:, 1].max(); x0 = allp[:, 0].min()
    f = (DIGIT_BOT - DIGIT_TOP) / max(bot - top, 1e-6)
    return [np.column_stack([(s[:, 0] - x0) * f, (s[:, 1] - top) * f + DIGIT_TOP, s[:, 2]]) for s in strokes]


def collect(prefixes):
    out = {c: [] for c in "0123456789"}
    for prefix in prefixes:
        pages = load(prefix)
        for pi, pg in enumerate(pages):
            boxes = [w for w in pg["words"] if len(w["text"].strip(".,;:()[]")) == 1 and w["text"].strip(".,;:()[]").isdigit()]
            if not boxes: continue
            owner = assign_strokes(pg["strokes"], boxes, pad=0.5)
            for bi, w in enumerate(boxes):
                idx = np.flatnonzero(owner == bi)
                if not 1 <= len(idx) <= 2: continue
                strokes = [pg["strokes"][i] for i in idx]
                allp = np.vstack([s[:, :2] for s in strokes]); h = np.ptp(allp[:, 1]); wd = np.ptp(allp[:, 0])
                if h < 3 or wd / h > 1.3 or h / max(wd, 1e-6) > 6: continue   # too small, too wide, or a bare stem
                d = w["text"].strip(".,;:()[]")
                if d not in out: continue   # circled digits and other oddities
                pieces = normalise(strokes); allq = np.vstack([p[:, :2] for p in pieces])
                out[d].append({"text": d, "pieces": pieces, "width": float(allq[:, 0].max()), "height": float(np.ptp(allq[:, 1])),
                               "src": (prefix, pi, "ocr", bi), "single": True, "ocr": True})
    return out


def filter_by_seed(cands, seeds, keep_frac=0.6):
    """keep the candidates closest in shape to the sheet seed(s); returns sorted best-first"""
    out = {}
    for c, insts in cands.items():
        sd = [shape(u["pieces"]) for u in seeds.get(c, [])]
        if not sd or not insts: out[c] = []; continue
        scored = []
        for u in insts:
            f = shape(u["pieces"]); dist = min(np.linalg.norm(f[:-1] - s[:-1]) / np.sqrt(len(f)) + (0.3 if f[-1] != s[-1] else 0) for s in sd)
            scored.append((dist, u))
        scored.sort(key=lambda t: t[0]); n = max(3, int(len(scored) * keep_frac))
        out[c] = [dict(u, dist=float(d)) for d, u in scored[:n]]
    return out


if __name__ == "__main__":
    prefixes = [p[:-5] for p in sorted(glob.glob("data/derived/*.json")) if not p.endswith(".words.json")]
    cands = collect(prefixes); print("OCR digit candidates:", {c: len(v) for c, v in cands.items()})
    seeds = pickle.load(open("data/derived/symbols.pkl", "rb"))
    kept = filter_by_seed(cands, seeds, keep_frac=0.7)
    # second pass: the best notebook instances become the templates (they are closer to how the writer writes digits in notes)
    tmpl = {c: v[:6] for c, v in kept.items()}
    kept = filter_by_seed(cands, {c: tmpl[c] + seeds.get(c, []) for c in tmpl}, keep_frac=0.6)
    print("kept after two passes:", {c: len(v) for c, v in kept.items()})
    pickle.dump(kept, open("data/derived/digits_ocr.pkl", "wb"))
