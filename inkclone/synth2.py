"""Longest-unit synthesis: whole word if available, else joined chunks + letters,
placed left to right with measured gaps. Units are never joined to each other, so
every seam sits at a real pen-up."""
import pickle, random, re
import numpy as np


class Synth2:
    def __init__(self, units="data/derived/units.pkl", bank="data/derived/bank.pkl", seed=None):
        u = pickle.load(open(units, "rb")); self.W, self.C, self.L = u["words"], u["chunks"], u["letters"]
        self.letter_h = u.get("letter_h", {}); self.slant_target = None; self.pen_target = None
        self.stats = pickle.load(open(bank, "rb"))["stats"]; self.rng = random.Random(seed)
        self.lower = {}
        for t, insts in self.W.items(): self.lower.setdefault(t.lower(), []).extend(insts)

    def gap(self, a, b):
        g = self.stats["gap_by_pair"].get(a + b, self.stats["gap_median"]); lo, hi = self.stats["gap_iqr"]
        return max(0.0, g + self.rng.uniform(lo - g, hi - g) * 0.5)

    def tile(self, text):
        """-> list of units covering text (chunks preferred, longest first with some randomness)."""
        out, i = [], 0
        while i < len(text):
            opts = [L for L in range(min(6, len(text) - i), 1, -1) if self.C.get(text[i:i+L])]
            if opts and self.rng.random() < 0.85:
                L = opts[0] if len(opts) == 1 or self.rng.random() < 0.7 else self.rng.choice(opts)
                out.append(self.pick_by_quality(self.C[text[i:i+L]])); i += L
            elif text[i] in self.L and self.L[text[i]]:
                pool = self.L[text[i]]; whole = [u for u in pool if u.get("whole")]
                if whole and (len(whole) >= 3 or self.rng.random() < 0.8): pool = whole
                out.append(self.pick_by_quality(pool)); i += 1
            else:
                out.append(None); i += 1
        return out

    def conform(self, pieces, u):
        """Affine-only corrections so every unit on a line agrees: pull single letters toward their
        class height, shear to the line's slant, and scale pen width to the document's pen."""
        pieces = [p.copy() for p in pieces]
        t = u.get("text", "")
        if len(t) == 1 and t.isalpha() and t in self.letter_h and u.get("height"):
            f = (self.letter_h[t] / u["height"]) ** 0.5; f = min(1.25, max(0.8, f))
            for p in pieces: p[:, :2] *= f
            u = dict(u, width=u["width"] * f)
        if self.slant_target is not None and u.get("slant") is not None:
            d = max(-12.0, min(12.0, self.slant_target - u["slant"]))
            k = np.tan(np.radians(self.slant_target)) - np.tan(np.radians(self.slant_target - d))
            for p in pieces: p[:, 0] += p[:, 1] * k
            x0 = min(p[:, 0].min() for p in pieces)
            for p in pieces: p[:, 0] -= x0
            u = dict(u, width=max(p[:, 0].max() for p in pieces))
        if self.pen_target is not None and u.get("pen"):
            f = min(2.0, max(0.5, self.pen_target / u["pen"]))
            for p in pieces: p[:, 2] *= f
        return pieces, u["width"]

    def pick_by_quality(self, pool, tau=0.25):
        """sample an instance, favouring low template-mismatch scores (still random, never argmin)."""
        sc = np.array([u.get("score") if u.get("score") is not None else 1.0 for u in pool], float)
        w = np.exp(-(sc - sc.min()) / tau); w /= w.sum()
        return pool[int(np.random.default_rng(self.rng.getrandbits(32)).choice(len(pool), p=w))]

    def word(self, text, used):
        src = "word"
        cands = self.W.get(text) or (self.lower.get(text.lower()) if text.islower() else None)
        if cands:
            fresh = [c for c in cands if id(c) not in used] or cands
            u = self.pick_by_quality(fresh); used.add(id(u)); pieces, w = self.conform(u["pieces"], u); return pieces, w, src
        strokes, x, prev = [], 0.0, None
        for u in self.tile(text):
            if u is None: x += 0.8; prev = None; continue
            if prev is not None: x += self.gap(prev[-1], u["text"][0])
            pieces, w = self.conform(u["pieces"], u)
            for p in pieces: q = p.copy(); q[:, 0] += x; strokes.append(q)
            x += w; prev = u["text"]
        return strokes, x, "units"

    def line(self, text, xh_pt=None):
        xh = xh_pt or self.stats["xheight_pt_median"]; wg = self.stats.get("word_gap_xh_median", 2.0)
        out, x, used, how = [], 0.0, set(), []
        for tok in re.findall(r"\S+", text):
            core = re.match(r"^([A-Za-z]+)(.*)$", tok)
            word, tail = (core.group(1), core.group(2)) if core else (tok, "")
            s, width, src = self.word(word, used); how.append(src)
            jit = self.rng.uniform(0.94, 1.06)
            for p in s: q = p * [xh * jit, xh * jit, 1.0]; q[:, 0] += x; out.append(q)
            x += width * xh * jit
            if tail:  # punctuation: use letter bank if we have it
                for ch in tail:
                    if ch in self.L and self.L[ch]:
                        u = self.rng.choice(self.L[ch]); x += 0.15 * xh
                        for p in u["pieces"]: q = p * [xh, xh, 1.0]; q[:, 0] += x; out.append(q)
                        x += u["width"] * xh
            x += wg * xh * self.rng.uniform(0.7, 1.3)
        return out, x, how
