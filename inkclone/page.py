"""Lay out a parsed typed document as handwritten pages.

Conventions copied from the writer's HW pages: numbered section headers at a larger
size, pseudocode kept with its indentation, prose wrapped to the page width. Letters
and words come from Synth2 (whole words > chunks > letters), symbols and digits from
the symbol bank, sub/superscripts scaled down and shifted. Pages are GoodNotes-sized.
"""
import re, pickle, random, collections
import numpy as np
from .synth2 import Synth2
from .extract import render_strokes

PAGE_W, PAGE_H = 455.0, 588.0
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 18.0, 14.0, 16.0, 20.0


class Page:
    def __init__(self, seed=0, body_xh=3.75, header_xh=6.6, two_column=True, note_scale=0.8, col_split=0.52):
        self.s = Synth2(seed=seed); self.rng = random.Random(seed)
        self.sym = pickle.load(open("data/derived/symbols_final.pkl", "rb"))
        self.body_xh, self.header_xh = body_xh, header_xh
        self.two_column, self.note_scale, self.col_split = two_column, note_scale, col_split
        self.missing = collections.Counter()
        self.s.pen_target = self.s.stats.get("pen_width_pt", 0.7)  # the writer's measured pen width in pt
        self.doc_slant = max(-9.0, min(0.0, self.rng.gauss(self.s.stats.get("slant_deg_median", -4.4), 1.5)))
        self.drift = self.s.stats.get("drift", {"pitch_ratio_sd": 0.12, "baseline_slope_sd": 0.01, "baseline_slope_median": 0.0, "margin_sd_pt": 2.0})

    def pitch(self):
        return max(0.92, min(1.18, self.rng.gauss(1.0, 0.25 * self.drift["pitch_ratio_sd"])))  # measured sd includes paragraph gaps; keep it gentle

    def margin_jitter(self):
        return self.rng.gauss(0, 0.35 * self.drift["margin_sd_pt"])  # measured sd includes indentation noise; use a third

    # ---- glyph sources -------------------------------------------------------------
    PUNCT_FROM_PROSE = set("'?!")  # prose-cut punctuation is dirty; ':' and ';' are composed from the writer's own periods/commas

    COMPOSITE = {"⇒": "=>", "→": "->", "⇔": "<=>", "↔": "<->", "≠": "=/"}

    def mirror(self, u):
        """horizontal mirror of a glyph instance (for '>' from '<')."""
        pieces = [p.copy() for p in u["pieces"]]
        for p in pieces: p[:, 0] = u["width"] - p[:, 0]
        return dict(u, pieces=pieces)

    def n_real(self, c):
        return sum(1 for u in self.sym.get(c, []) if not u.get("synthetic"))

    def symbol(self, c):
        if c == ">" and not self.sym.get(">") and self.sym.get("<"): return self.mirror(self.rng.choice(self.sym["<"]))
        if c in ":;" and len(self.sym.get(".", [])) >= 3:
            top = self.rng.choice(self.sym["."]); bot = self.rng.choice(self.sym["."] if c == ":" else (self.sym.get(",") or self.sym["."]))
            pt = [p.copy() for p in top["pieces"]]; pb = [p.copy() for p in bot["pieces"]]
            for p in pt: p[:, 1] -= 0.75   # upper dot at mid x-height
            w = max(max(p[:, 0].max() for p in pt), max(p[:, 0].max() for p in pb))
            return {"text": c, "pieces": pt + pb, "width": float(w), "height": 1.0, "slant": top.get("slant"), "pen": top.get("pen")}
        if c == "/" and self.n_real("/") < 3 and len(self.sym.get("1", [])) >= 5:
            one = self.rng.choice(self.sym["1"][:40]); st = max(one["pieces"], key=len).copy()
            st[:, 1] -= st[:, 1].mean(); st[:, 1] *= 1.9 / max(st[:, 1].max() - st[:, 1].min(), 1e-6)  # ~1.9 xh tall
            st[:, 0] = st[:, 0] - st[:, 0].mean() - st[:, 1] * 0.45  # lean it: top to the right
            st[:, 1] += -0.75; st[:, 0] -= st[:, 0].min()
            return {"text": "/", "pieces": [st], "width": float(st[:, 0].max()), "height": 1.9, "slant": one.get("slant"), "pen": one.get("pen")}
        if c == "-" and self.n_real("-") < 3 and len(self.sym.get("=", [])) >= 5:
            eq = self.rng.choice(self.sym["="][:40]); bar = max(eq["pieces"], key=len).copy()
            bar[:, 1] -= bar[:, 1].mean() - (-0.45)  # a hyphen sits a little under mid x-height
            bar[:, 0] -= bar[:, 0].min()
            return {"text": "-", "pieces": [bar], "width": float(bar[:, 0].max()), "height": float(bar[:, 1].max() - bar[:, 1].min()), "slant": eq.get("slant"), "pen": eq.get("pen")}
        if c == ")" and len(self.sym.get("(", [])) >= 5: return self.mirror(self.rng.choice(self.sym["("][:30]))
        if c == "]" and self.n_real("]") < 3 and len(self.sym.get("[", [])) >= 5: return self.mirror(self.rng.choice(self.sym["["][:20]))
        if c == "0" and not self.sym.get("0") and self.s.L.get("o"): return self.s.pick_by_quality([u for u in self.s.L["o"] if u.get("whole")] or self.s.L["o"])
        if c in self.PUNCT_FROM_PROSE and self.s.L.get(c):
            return self.s.pick_by_quality(self.s.L[c])
        pool = self.sym.get(c)
        if not pool: self.missing[c] += 1; return None
        n_good = max(2, len(pool) // 2)  # lists are best-first
        return self.rng.choice(pool[:n_good]) if len(pool) >= 4 else self.rng.choice(pool)

    def composite(self, c):
        """Build ⇒ → ⇔ from parts laid tightly left to right; ≠ is '=' with the slash centred. -> (strokes, width) or None."""
        parts = self.COMPOSITE.get(c)
        if not parts: return None
        if c == "≠":
            eq, sl = self.symbol("="), self.symbol("/")
            if eq is None or sl is None: return None
            strokes = [p.copy() for p in eq["pieces"]]; dx = (eq["width"] - sl["width"]) / 2
            for p in sl["pieces"]: q = p.copy(); q[:, 0] += dx; strokes.append(q)
            return strokes, eq["width"]
        strokes, x = [], 0.0
        for k, pc in enumerate(parts):
            u = self.symbol(pc)
            if u is None: return None
            dx = -0.35 if k > 0 else 0.0  # overlap the arrow head onto the shaft
            for p in u["pieces"]: q = p.copy(); q[:, 0] += x + dx; strokes.append(q)
            x += u["width"] + dx
        return strokes, x

    def fraction(self, tok, used):
        """frac{num}{den} -> numerator above a bar, denominator below. Returns (strokes, width) or None."""
        m = re.match(r"^frac\{(.*?)\}\{(.*?)\}(.*)$", tok)
        if not m: return None
        num, den, tail = m.group(1).replace("~", " "), m.group(2).replace("~", " "), m.group(3)
        parts = []
        for text, dy in ((num, -1.75), (den, 1.35)):
            st, x = [], 0.0
            for k, t in enumerate(text.split()):
                ts, tw = self.token(t, used)
                for p_ in ts: q = p_.copy(); q[:, 0] += x; st.append(q)
                x += tw + 0.6
            x -= 0.6
            for p_ in st: p_[:, :2] *= 0.8; p_[:, 1] += dy * 0.8
            parts.append((st, x * 0.8))
        w = max(parts[0][1], parts[1][1]) + 0.4
        strokes = []
        for st, pw in parts:
            for p_ in st: p_[:, 0] += (w - pw) / 2; strokes.append(p_)
        bar = self.symbol("-")
        if bar:
            b = max(bar["pieces"], key=len).copy(); b[:, 0] = (b[:, 0] - b[:, 0].min()) / max(b[:, 0].max() - b[:, 0].min(), 1e-6) * w
            b[:, 1] = b[:, 1] - b[:, 1].mean() - 0.45; strokes.append(b)
        if tail:
            ts, tw = self.token(tail, used)
            for p_ in ts: q = p_.copy(); q[:, 0] += w + 0.2; strokes.append(q)
            w += tw + 0.2
        return strokes, w

    def token(self, tok, used):
        """-> (strokes in xh units, width in xh). Handles letters, digits, symbols, _ and ^, frac{}{}."""
        if tok.startswith("frac{"):
            fr = self.fraction(tok, used)
            if fr: return fr
        strokes, x, i = [], 0.0, 0
        mode, group = None, False  # sub/superscript state
        while i < len(tok):
            c = tok[i]
            if c in "_^": mode = c; i += 1; group = tok[i:i+1] == "{"; i += 1 if group else 0; continue
            if c == "}" and group: mode, group = None, False; i += 1; continue
            scale, dy = (0.65, 0.55) if mode == "_" else (0.65, -1.0) if mode == "^" else (1.0, 0.0)
            if c.isalpha():
                j = i
                while j < len(tok) and tok[j].isalpha(): j += 1
                run = tok[i:j]; st, w, _ = self.s.word(run, used); i = j
            else:
                i += 1
                comp = self.composite(c) if c in self.COMPOSITE else None
                if comp: st, w = comp
                else:
                    u = self.symbol(c)
                    if u is None: x += 0.8 * scale; continue
                    st, w = self.s.conform(u["pieces"], u)
            if strokes: x += 0.22 * scale
            for p in st:
                q = p.copy(); q[:, :2] *= scale; q[:, 0] += x; q[:, 1] += dy; strokes.append(q)
            x += w * scale
            if mode and not group: mode = None
        return strokes, x

    # ---- layout --------------------------------------------------------------------
    def line_strokes(self, text, xh, x0, baseline, used, max_w=None):
        """Lay tokens along a baseline. Returns (strokes in pt, leftover tokens if wrapped)."""
        x0 = x0 + self.margin_jitter()
        out, x = [], x0; toks = text.split(); wg = self.s.stats.get("word_gap_xh_median", 2.0)
        self.s.slant_target = max(-12.0, min(2.0, self.doc_slant + self.rng.gauss(0, 1.8)))  # one slant per line, drifting around the document's
        xh = xh * self.rng.uniform(0.95, 1.05)  # size mood per line
        slope = max(-0.012, min(0.012, self.rng.gauss(self.drift["baseline_slope_median"], 0.5 * self.drift["baseline_slope_sd"])))  # baseline wander, capped at ~5pt per line
        for n, tok in enumerate(toks):
            st, w = self.token(tok, used); jit = self.rng.uniform(0.95, 1.05)
            if max_w and n > 0 and x + w * xh * jit > x0 + max_w: return out, toks[n:]
            by = baseline + self.rng.uniform(-0.12, 0.12) * xh + slope * (x - x0)
            for p in st: q = p.copy(); q[:, :2] *= xh * jit; q[:, 0] += x; q[:, 1] += by; out.append(q)
            x += w * xh * jit + wg * xh * self.rng.uniform(0.7, 1.25)
        return out, []

    def sections(self, lines, include_title=False):
        """Group parsed lines into sections: header, intro prose (before code), code lines, notes (prose after code)."""
        secs, cur = [], None
        for ln in lines:
            if ln["kind"] == "meta" or (ln["kind"] == "title" and not include_title): continue
            if ln["kind"] in ("header", "title") or cur is None:
                cur = {"header": ln["text"] if ln["kind"] in ("header", "title") else None, "intro": [], "code": [], "notes": []}; secs.append(cur)
                if ln["kind"] not in ("header", "title"): cur[("code" if ln["kind"] == "code" else "intro")].append(ln)
                continue
            if ln["kind"] == "code": cur["code"].append(ln)
            elif cur["code"]: cur["notes"].append(ln)
            else: cur["intro"].append(ln)  # math lines travel with the prose and are laid out as single lines
        return secs

    def prose_block(self, text, xh, x0, y, max_w, used, math=False):
        """wrap a paragraph; returns (strokes, new y). Math display lines are indented and never wrapped."""
        if math:
            st, _ = self.line_strokes(text, xh, x0 + 6 * xh, y + 1.8 * xh, used); return st, y + 4.2 * xh * self.pitch()
        out, rest = [], text
        while rest:
            st, left = self.line_strokes(rest, xh, x0, y + 1.8 * xh, used, max_w=max_w); out += st; y += 4.0 * xh * self.pitch()
            rest = " ".join(left) if left else ""
        return out, y

    def layout(self, lines, include_title=False):
        pages, cur, y = [], [], MARGIN_T; used = set()
        full_w = PAGE_W - MARGIN_L - MARGIN_R
        def newpage():
            nonlocal cur, y
            if cur: pages.append(cur)
            cur, y = [], MARGIN_T
        def need(h):
            if y + h > PAGE_H - MARGIN_B: newpage()
        xh = self.body_xh
        for si, sec in enumerate(self.sections(lines, include_title)):
            if sec["header"]:
                if si: y += 2.0 * xh
                need(3.2 * self.header_xh + 8 * xh)
                st, _ = self.line_strokes(sec["header"], self.header_xh, MARGIN_L, y + 2.2 * self.header_xh, used); cur += st; y += 3.6 * self.header_xh
            for ln in sec["intro"]:
                need(4 * xh); st, y = self.prose_block(ln["text"], xh, MARGIN_L, y, full_w, used, math=(ln["kind"] == "math")); cur += st
            if sec["code"]:
                need(3.6 * xh * min(len(sec["code"]), 4))
                y_top = y; code_strokes = []
                for ln in sec["code"]:
                    need(3.6 * xh)
                    if y == MARGIN_T: y_top = y; code_strokes = []
                    x0 = MARGIN_L + 3.2 * xh * ln["indent"] + (0 if ln["indent"] == 0 else 0.6 * xh)
                    st, _ = self.line_strokes(ln["text"], xh, x0, y + 1.8 * xh, used); cur += st; code_strokes += st; y += 3.9 * xh * self.pitch()
                y_code_end = y
                if self.two_column and sec["notes"] and code_strokes:
                    code_right = max(p[:, 0].max() for p in code_strokes)
                    nx0 = max(code_right + 2.5 * xh, MARGIN_L + 0.42 * full_w); nw = PAGE_W - MARGIN_R - nx0; nxh = xh * self.note_scale
                    if nw >= 0.30 * full_w:
                        yn = y_top + 0.6 * xh; pending = list(sec["notes"])
                        while pending and yn < PAGE_H - MARGIN_B - 4 * nxh:
                            ln = pending.pop(0); st, yn = self.prose_block(ln["text"], nxh, nx0, yn, nw, used); cur += st; yn += 0.8 * nxh
                        y = max(y_code_end, yn)
                        sec = dict(sec, notes=pending)  # anything that did not fit goes below, full width
            for ln in sec["notes"]:
                need(4 * xh); st, y = self.prose_block(ln["text"], xh, MARGIN_L, y, full_w, used, math=(ln["kind"] == "math")); cur += st
        newpage(); return pages

    def render(self, pages, out_prefix, dpi=300, vector=True):
        """Main output is vector ink (<prefix>.pdf). PNG previews are rasterised from the vector pages."""
        import pymupdf
        vdoc = pymupdf.open()
        for st in pages:
            write_vector_page(vdoc.new_page(width=PAGE_W, height=PAGE_H), st)
        vdoc.save(out_prefix + ".pdf")
        for n, pg in enumerate(pymupdf.open(out_prefix + ".pdf")):
            pg.get_pixmap(dpi=dpi).save(f"{out_prefix}_p{n+1}.png")
        return len(pages)


def clean_centerline(P, w, step=0.12, window_pt=0.6):
    """Remove the sub-pen-width zig-zag the extractor leaves where ribbon pieces overlap:
    resample by arc length, then smooth x/y with a window about one pen width long."""
    if len(P) < 4:
        return P, w
    # drop backtracks: a sample within `r` of any of the last `look` kept samples is a step back onto
    # path already drawn (overlapping ribbon pieces), not new ink
    r, look = 0.2, 15; keep = [0]
    for i in range(1, len(P)):
        recent = P[keep[-look:]]
        if np.min(np.linalg.norm(recent - P[i], axis=1)) >= r or i == len(P) - 1:
            keep.append(i)
    P, w = P[keep], w[keep]
    d = np.linalg.norm(np.diff(P, axis=0), axis=1); cum = np.r_[0, np.cumsum(d)]
    if cum[-1] < step * 2:
        return P, w
    t = np.arange(0, cum[-1], step); t = np.r_[t, cum[-1]]
    Q = np.column_stack([np.interp(t, cum, P[:, 0]), np.interp(t, cum, P[:, 1])]); wq = np.interp(t, cum, w)
    k = max(1, int(round(window_pt / step)) | 1)  # odd window
    if len(Q) > k:
        pad = k // 2; ker = np.ones(k) / k
        xs = np.convolve(np.r_[np.full(pad, Q[0, 0]), Q[:, 0], np.full(pad, Q[-1, 0])], ker, mode="valid")
        ys = np.convolve(np.r_[np.full(pad, Q[0, 1]), Q[:, 1], np.full(pad, Q[-1, 1])], ker, mode="valid")
        Q = np.column_stack([xs, ys]); Q[0], Q[-1] = P[0], P[-1]  # keep the true end points
    return Q, wq


def write_vector_page(page, strokes, color=(0.12, 0.11, 0.11)):
    """Each stroke is drawn as stroked polylines with round caps and joins (the PDF renderer does
    the geometry, so no self-intersecting outlines). The pen width varies along the stroke, so a
    stroke is split into short runs of near-constant width; runs overlap by one point."""
    import pymupdf
    shape = page.new_shape()
    for s in strokes:
        if len(s) < 2:
            continue
        P, w = clean_centerline(s[:, :2], np.maximum(s[:, 2], 0.15))
        # one width per stroke (the pen's pressure variation within a stroke is ~10% and the
        # round-join scalloping from per-run widths was far more visible than that variation)
        shape.draw_polyline([pymupdf.Point(float(x), float(y)) for x, y in P])
        shape.finish(width=float(np.median(w)), color=color, lineCap=1, lineJoin=1)
    shape.commit()
