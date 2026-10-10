"""Longest-unit synthesis: whole word if available, else joined chunks + letters,
placed left to right with measured gaps. Units are never joined to each other, so
every seam sits at a real pen-up."""
import pickle, random, re
import numpy as np


class Synth2:
    def __init__(self, units="data/derived/units.pkl", bank="data/derived/bank.pkl", seed=None):
        u = pickle.load(open(units, "rb")); self.W, self.C, self.L = u["words"], u["chunks"], u["letters"]
        self.letter_h = u.get("letter_h", {}); self.letter_ext = u.get("letter_ext", {}); self.slant_target = None; self.pen_target = None
        self.stats = pickle.load(open(bank, "rb"))["stats"]; self.rng = random.Random(seed)
        self.lower = {}
        for t, insts in self.W.items(): self.lower.setdefault(t.lower(), []).extend(insts)
        pc, jr = self.stats["pair_count"], self.stats["join_rate"]; self.out_of, self.into = {}, {}
        for p_, n in pc.items():
            a = self.out_of.setdefault(p_[0], [0, 0]); a[0] += jr[p_] * n; a[1] += n
            b = self.into.setdefault(p_[1], [0, 0]); b[0] += jr[p_] * n; b[1] += n
        self.join_enabled = False  # joined-variant letters are cut mid-stroke and bypass curation; off until cuts are trustworthy
        self.use_whole_words = False  # everything is assembled letter by letter from the curated bank; no word-level pasting

    def join_prob(self, a, b):
        pc, jr = self.stats["pair_count"], self.stats["join_rate"]
        if pc.get(a + b, 0) >= 4: return jr[a + b]
        g = self.stats["global_join_rate"]
        pa = self.out_of[a][0] / self.out_of[a][1] if self.out_of.get(a, [0, 0])[1] >= 10 else g
        pb = self.into[b][0] / self.into[b][1] if self.into.get(b, [0, 0])[1] >= 10 else g
        return min(0.9, pa * pb / g) if g else 0.0

    def joined_variant(self, c, prev=False, nxt=False):
        """a well-scored single-letter instance of c with the requested cut flags, or None."""
        pool = [u for u in self.L.get(c, []) if (not prev or u.get("joined_prev")) and (not nxt or u.get("joined_next")) and u.get("score") is not None]
        if len(pool) < 2: return None
        pool.sort(key=lambda u: u["score"]); return self.rng.choice(pool[:max(2, len(pool) // 3)])

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
        if u.get("pen") is None:   # harvested symbols carry no pen/slant record: measure them
            u = dict(u, pen=float(np.median(np.concatenate([p[:, 2] for p in pieces]))))
        if u.get("slant") is None:
            from .bank import slant_deg
            u = dict(u, slant=slant_deg(np.vstack([p[:, :2] for p in pieces])))
        ext = [self.letter_ext[c] for c in t if c in self.letter_ext]
        if ext and (t.isalpha() or t.isdigit() or t in "()[]{}|"):
            exp_top, exp_bot = min(e[0] for e in ext), max(e[1] for e in ext)
            top = min(p[:, 1].min() for p in pieces); bot = max(p[:, 1].max() for p in pieces)
            if bot - top > 1e-3:
                power = 0.75 if len(t) <= 3 else 0.5  # pull chunks hard, whole words gently
                f = ((exp_bot - exp_top) / (bot - top)) ** power
                f = min(2.2, max(0.45, f)) if t.isdigit() else min(1.4, max(0.7, f))  # digit seeds come from rough symbol-line frames
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
            f = min(3.0, max(0.3, self.pen_target / u["pen"]))
            for p in pieces: p[:, 2] *= f
        return pieces, u["width"]

    def pick_by_quality(self, pool, tau=0.25):
        """sample an instance, favouring low template-mismatch scores (still random, never argmin)."""
        sc = np.array([u.get("score") if u.get("score") is not None else 1.0 for u in pool], float)
        w = np.exp(-(sc - sc.min()) / tau); w /= w.sum()
        return pool[int(np.random.default_rng(self.rng.getrandbits(32)).choice(len(pool), p=w))]

    def word(self, text, used):
        src = "word"
        cands = None if not self.use_whole_words else self.W.get(text)
        if self.use_whole_words and not cands and text.islower():  # lowercase token: only lowercase instances (never a capitalised one)
            cands = [u for u in self.lower.get(text.lower(), []) if u["text"].islower()] or None
        if cands:
            fresh = [c for c in cands if id(c) not in used] or cands  # variants make a fresh one almost always available
            u = self.pick_by_quality(fresh); used.add(id(u)); pieces, w = self.conform(u["pieces"], u); return pieces, w, src
        units = self.tile(text)
        # decide joins between adjacent single-letter units at the writer's measured pair rates
        joins = [False] * max(0, len(units) - 1)
        if self.join_enabled:
            for k in range(len(units) - 1):
                a, b = units[k], units[k + 1]
                if a and b and len(a["text"]) == 1 and len(b["text"]) == 1 and a["text"].isalpha() and b["text"].isalpha():
                    if self.rng.random() < self.join_prob(a["text"], b["text"]):
                        a2 = self.joined_variant(a["text"], nxt=True); b2 = self.joined_variant(b["text"], prev=True)
                        if a2 is not None and b2 is not None: units[k], units[k + 1] = a2, b2; joins[k] = True
        strokes, x, prev = [], 0.0, None; prev_exit = None; prev_x0 = None
        for k, u in enumerate(units):
            if u is None: x += 0.8; prev = None; prev_exit = None; continue
            pieces, w = self.conform(u["pieces"], u)
            if k > 0 and joins[k - 1] and prev_exit is not None and strokes:
                ex, ey = prev_exit; nx, ny = pieces[0][0, :2]
                x0 = prev_x0 + ex - nx; x0 = max(x0, x - 0.35 * w); dy = max(-0.25, min(0.25, ey - ny))
                for p in pieces: p[:, 0] += x0; p[:, 1] += dy
                last = strokes[-1]; a_, b_ = last[-1, :2], pieces[0][0, :2]
                if np.hypot(*(b_ - a_)) > 0.02:
                    t = np.linspace(0, 1, 3)[1:-1, None]; mid = last[-1] + (pieces[0][0] - last[-1]) * t; last = np.vstack([last, mid])
                strokes[-1] = np.vstack([last, pieces[0]]); strokes.extend(pieces[1:])
            else:
                x0 = x + (self.gap(prev[-1], u["text"][0]) if prev is not None else 0.0)
                for p in pieces: p[:, 0] += x0
                strokes.extend(pieces)
            x = x0 + w; prev = u["text"]; prev_x0 = x0
            prev_exit = tuple(pieces[0][-1, :2]) if (k < len(units) - 1 and joins[k]) else None
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
