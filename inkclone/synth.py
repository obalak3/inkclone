"""Synthesise handwriting by selecting real character instances from the bank.

For each adjacent letter pair, decide join/no-join by sampling the measured join rate
for that pair (falling back to letter-level and then global rates). Pick an instance
whose recorded join flags match, place it, and when joined, continue the previous
pen stroke straight into it with a short connector so the result is one stroke.
"""
import pickle, random, json, glob
import numpy as np

XH_LETTERS = set("acemnorsuvwxz")


class Synth:
    def __init__(self, bank_path="data/derived/bank.pkl", seed=None):
        d = pickle.load(open(bank_path, "rb")); self.bank, self.stats = d["bank"], d["stats"]
        self.rng = random.Random(seed)
        pc, jr = self.stats["pair_count"], self.stats["join_rate"]
        self.out_of, self.into = {}, {}
        for p, n in pc.items():
            a = self.out_of.setdefault(p[0], [0, 0]); a[0] += jr[p] * n; a[1] += n
            b = self.into.setdefault(p[1], [0, 0]); b[0] += jr[p] * n; b[1] += n

    def join_prob(self, a, b):
        pc, jr = self.stats["pair_count"], self.stats["join_rate"]
        if pc.get(a + b, 0) >= 4: return jr[a + b]
        g = self.stats["global_join_rate"]
        pa = self.out_of[a][0] / self.out_of[a][1] if self.out_of.get(a, [0, 0])[1] >= 10 else g
        pb = self.into[b][0] / self.into[b][1] if self.into.get(b, [0, 0])[1] >= 10 else g
        return min(0.95, pa * pb / g) if g else 0.0  # independent-ish combination

    def pick(self, c, want_prev, want_next, used):
        insts = self.bank.get(c)
        if not insts: return None
        for relax in range(3):
            cand = [i for i in insts if (relax >= 1 or i["joined_prev"] == want_prev) and (relax >= 2 or i["joined_next"] == want_next)]
            cand2 = [i for i in cand if id(i) not in used] or cand
            if cand2: return self.rng.choice(cand2)
        return self.rng.choice(insts)

    def gap(self, a, b):
        g = self.stats["gap_by_pair"].get(a + b)
        lo, hi = self.stats["gap_iqr"]
        base = g if g is not None else self.stats["gap_median"]
        return base + self.rng.uniform(lo - base, hi - base) * 0.5

    def word(self, text, used=None):
        """-> list of strokes in normalised units (x from 0, baseline y=0), and width."""
        used = used if used is not None else set()
        chars = list(text)
        joins = [self.rng.random() < self.join_prob(a, b) for a, b in zip(chars, chars[1:])]
        joins = [j and self.bank.get(a) and any(i['joined_next'] and i['advance'] is not None for i in self.bank[a]) for j, a in zip(joins, chars)]
        strokes, x = [], 0.0
        prev = None
        for k, c in enumerate(chars):
            jp = k > 0 and joins[k-1]; jn = k < len(chars) - 1 and joins[k]
            inst = self.pick(c, jp, jn, used)
            if inst is None:  # unknown glyph: leave a space
                x += 1.0; prev = None; continue
            used.add(id(inst))
            pieces = [p.copy() for p in inst["pieces"]]
            if jp and prev is not None and strokes and prev["advance"] is not None:
                # continue the previous pen stroke: put this letter's entry exactly on the previous exit
                ex, ey = prev["exit"]; nx, ny = inst["entry"]
                x0 = prev["x0"] + ex - nx
                x0 = min(max(x0, prev["x0"] + 0.6 * prev["advance"]), prev["x0"] + 1.4 * prev["advance"])
                dy = ey - ny; dy = max(-0.25, min(0.25, dy))
                for p in pieces: p[:, 0] += x0; p[:, 1] += dy
                last = strokes[-1]; a, b = last[-1], pieces[0][0]
                if np.hypot(*(b[:2] - a[:2])) > 0.02:
                    t = np.linspace(0, 1, 3)[1:-1, None]; last = np.vstack([last, a + (b - a) * t])
                strokes[-1] = np.vstack([last, pieces[0]]); strokes.extend(pieces[1:])
            else:
                x0 = x + (self.gap(chars[k-1], c) if k > 0 else 0.0)
                for p in pieces: p[:, 0] += x0
                strokes.extend(pieces)
            x = x0 + inst["width"]
            prev = {"exit": inst["exit"], "x0": x0, "width": inst["width"], "advance": inst["advance"]}
        return strokes, x

    def line(self, text, xh_pt=None, word_gap_xh=None):
        """Lay out a line of words. -> strokes in pt, baseline at y=0, x from 0."""
        xh = xh_pt or self.stats["xheight_pt_median"]
        wg = word_gap_xh if word_gap_xh is not None else self.stats.get("word_gap_xh_median", 1.0)
        out, x, used = [], 0.0, set()
        for w in text.split():
            s, width = self.word(w, used)
            jitter = self.rng.uniform(0.93, 1.07)
            for p in s:
                q = p * [xh * jitter, xh * jitter, 1.0]; q[:, 0] += x; out.append(q)
            x += width * xh * jitter + wg * xh * self.rng.uniform(0.75, 1.3)
        return out, x


def measure_word_gaps(derived_glob="data/derived/*.json", bank_path="data/derived/bank.pkl"):
    """Gap between consecutive OCR word boxes on a line, in x-heights. Stored into bank stats."""
    d = pickle.load(open(bank_path, "rb")); xh = d["stats"]["xheight_pt_median"]
    ratios, gaps = [], []
    for f in glob.glob(derived_glob):
        if f.endswith(".words.json"): continue
        for pg in json.load(open(f)):
            ws = sorted(pg["words"], key=lambda w: (round(w["bbox"][1] / 6), w["bbox"][0]))
            for a, b in zip(ws, ws[1:]):
                ay0, ay1 = a["bbox"][1], a["bbox"][3]; by0, by1 = b["bbox"][1], b["bbox"][3]
                h = min(ay1 - ay0, by1 - by0)
                if min(ay1, by1) - max(ay0, by0) > 0.5 * h and a["text"].isalpha() and b["text"].isalpha():
                    g = b["bbox"][0] - a["bbox"][2]
                    if 0 < g < 2.5 * h: gaps.append(g); ratios.append(h)
    gaps = np.array(gaps); H = np.median(ratios)
    # OCR box height ~ 2.3 x-heights for mixed-case words (ascender+descender); use measured xh instead
    d["stats"]["word_gap_xh_median"] = float(np.median(gaps) / xh); d["stats"]["word_gap_xh_iqr"] = [float(np.percentile(gaps, 25) / xh), float(np.percentile(gaps, 75) / xh)]
    pickle.dump(d, open(bank_path, "wb"))
    return d["stats"]["word_gap_xh_median"], d["stats"]["word_gap_xh_iqr"], len(gaps)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "."); from inkclone.extract import render_strokes
    text = sys.argv[1]; out = sys.argv[2]; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    s = Synth(seed=seed); strokes, width = s.line(text)
    xh = s.stats["xheight_pt_median"]
    img = render_strokes([p + [8, 3 * xh, 0] for p in strokes], (width + 16, 5 * xh), dpi=400)
    import pymupdf; pymupdf.Pixmap(pymupdf.csRGB, img.shape[1], img.shape[0], np.ascontiguousarray(img).tobytes(), False).save(out)
    print("wrote", out)
