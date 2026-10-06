"""Longest-unit synthesis: whole word if available, else joined chunks + letters,
placed left to right with measured gaps. Units are never joined to each other, so
every seam sits at a real pen-up."""
import pickle, random, re
import numpy as np


class Synth2:
    def __init__(self, units="data/derived/units.pkl", bank="data/derived/bank.pkl", seed=None):
        u = pickle.load(open(units, "rb")); self.W, self.C, self.L = u["words"], u["chunks"], u["letters"]
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
            opts = [L for L in range(min(6, len(text) - i), 1, -1) if text[i:i+L] in self.C]
            if opts and self.rng.random() < 0.85:
                L = opts[0] if len(opts) == 1 or self.rng.random() < 0.7 else self.rng.choice(opts)
                out.append(self.rng.choice(self.C[text[i:i+L]])); i += L
            elif text[i] in self.L and self.L[text[i]]:
                pool = self.L[text[i]]; whole = [u for u in pool if u.get("whole")]
                if whole and (len(whole) >= 3 or self.rng.random() < 0.8): pool = whole
                out.append(self.rng.choice(pool)); i += 1
            else:
                out.append(None); i += 1
        return out

    def word(self, text, used):
        src = "word"
        cands = self.W.get(text) or (self.lower.get(text.lower()) if text.islower() else None)
        if cands:
            fresh = [c for c in cands if id(c) not in used] or cands
            u = self.rng.choice(fresh); used.add(id(u)); return [p.copy() for p in u["pieces"]], u["width"], src
        strokes, x, prev = [], 0.0, None
        for u in self.tile(text):
            if u is None: x += 0.8; prev = None; continue
            if prev is not None: x += self.gap(prev[-1], u["text"][0])
            for p in u["pieces"]: q = p.copy(); q[:, 0] += x; strokes.append(q)
            x += u["width"]; prev = u["text"]
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
