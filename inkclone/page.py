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
    def __init__(self, seed=0, body_xh=4.2, header_xh=6.4):
        self.s = Synth2(seed=seed); self.rng = random.Random(seed)
        self.sym = pickle.load(open("data/derived/symbols_final.pkl", "rb"))
        self.body_xh, self.header_xh = body_xh, header_xh
        self.missing = collections.Counter()

    # ---- glyph sources -------------------------------------------------------------
    PUNCT_FROM_PROSE = set(";:'?!")  # commas/periods/hyphens from prose cuts are too dirty; use the sheet bank

    COMPOSITE = {"⇒": "=>", "→": "->", "⇔": "<=>", "↔": "<->", "≠": "=/"}

    def mirror(self, u):
        """horizontal mirror of a glyph instance (for '>' from '<')."""
        pieces = [p.copy() for p in u["pieces"]]
        for p in pieces: p[:, 0] = u["width"] - p[:, 0]
        return dict(u, pieces=pieces)

    def symbol(self, c):
        if c == ">" and not self.sym.get(">") and self.sym.get("<"): return self.mirror(self.rng.choice(self.sym["<"]))
        if c == ")" and len(self.sym.get(")", [])) < 5 and len(self.sym.get("(", [])) >= 5: return self.mirror(self.rng.choice(self.sym["("][:30]))
        if c == "]" and len(self.sym.get("]", [])) < 5 and len(self.sym.get("[", [])) >= 5: return self.mirror(self.rng.choice(self.sym["["][:20]))
        if c == "0" and not self.sym.get("0") and self.s.L.get("o"): return self.s.pick_by_quality([u for u in self.s.L["o"] if u.get("whole")] or self.s.L["o"])
        if c in self.PUNCT_FROM_PROSE and self.s.L.get(c):
            return self.s.pick_by_quality(self.s.L[c])
        pool = self.sym.get(c)
        if not pool: self.missing[c] += 1; return None
        n_good = max(2, len(pool) // 2)  # lists are best-first
        return self.rng.choice(pool[:n_good]) if len(pool) >= 4 else self.rng.choice(pool)

    def composite(self, c):
        """Build ⇒ → ⇔ from parts laid tightly left to right. -> (strokes, width) or None."""
        parts = self.COMPOSITE.get(c)
        if not parts: return None
        strokes, x = [], 0.0
        for k, pc in enumerate(parts):
            u = self.symbol(pc)
            if u is None: return None
            dx = -0.35 if k > 0 else 0.0  # overlap the arrow head onto the shaft
            for p in u["pieces"]: q = p.copy(); q[:, 0] += x + dx; strokes.append(q)
            x += u["width"] + dx
        return strokes, x

    def token(self, tok, used):
        """-> (strokes in xh units, width in xh). Handles letters, digits, symbols, _ and ^."""
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
                    st, w = [p.copy() for p in u["pieces"]], u["width"]
            if strokes: x += 0.22 * scale
            for p in st:
                q = p.copy(); q[:, :2] *= scale; q[:, 0] += x; q[:, 1] += dy; strokes.append(q)
            x += w * scale
            if mode and not group: mode = None
        return strokes, x

    # ---- layout --------------------------------------------------------------------
    def line_strokes(self, text, xh, x0, baseline, used, max_w=None):
        """Lay tokens along a baseline. Returns (strokes in pt, leftover tokens if wrapped)."""
        out, x = [], x0; toks = text.split(); wg = self.s.stats.get("word_gap_xh_median", 2.0)
        for n, tok in enumerate(toks):
            st, w = self.token(tok, used); jit = self.rng.uniform(0.95, 1.05)
            if max_w and n > 0 and x + w * xh * jit > x0 + max_w: return out, toks[n:]
            by = baseline + self.rng.uniform(-0.12, 0.12) * xh
            for p in st: q = p.copy(); q[:, :2] *= xh * jit; q[:, 0] += x; q[:, 1] += by; out.append(q)
            x += w * xh * jit + wg * xh * self.rng.uniform(0.7, 1.25)
        return out, []

    def layout(self, lines, include_title=False):
        pages, cur, y = [], [], MARGIN_T; used = set()
        def newpage():
            nonlocal cur, y
            if cur: pages.append(cur)
            cur, y = [], MARGIN_T
        def need(h):
            nonlocal y
            if y + h > PAGE_H - MARGIN_B: newpage()
        prev_kind = None
        for ln in lines:
            kind, text = ln["kind"], ln["text"]
            if kind == "meta" or (kind == "title" and not include_title): continue
            xh = self.header_xh if kind in ("header", "title") else self.body_xh
            if kind == "header":
                y += 1.6 * self.body_xh if prev_kind else 0; need(3.2 * xh)
                st, _ = self.line_strokes(text, xh, MARGIN_L, y + 2.2 * xh, used); cur += st; y += 3.4 * xh
            elif kind == "code":
                need(3 * xh); x0 = MARGIN_L + 3.2 * xh * ln["indent"] + (0 if ln["indent"] == 0 else 0.6 * xh)
                st, _ = self.line_strokes(text, xh, x0, y + 1.8 * xh, used); cur += st; y += 3.6 * xh
            else:  # prose: wrap
                rest = text; first = True
                while rest:
                    need(3 * xh)
                    st, left = self.line_strokes(rest, xh, MARGIN_L, y + 1.8 * xh, used, max_w=PAGE_W - MARGIN_L - MARGIN_R)
                    cur += st; y += 4.0 * xh; rest = " ".join(left) if left else ""
                if prev_kind == "code": pass
            prev_kind = kind
        newpage(); return pages

    def render(self, pages, out_prefix, dpi=300):
        import pymupdf
        doc = pymupdf.open()
        for n, st in enumerate(pages):
            img = render_strokes(st, (PAGE_W, PAGE_H), dpi=dpi)
            pix = pymupdf.Pixmap(pymupdf.csRGB, img.shape[1], img.shape[0], np.ascontiguousarray(img).tobytes(), False)
            pix.save(f"{out_prefix}_p{n+1}.png")
            pg = doc.new_page(width=PAGE_W, height=PAGE_H); pg.insert_image(pg.rect, pixmap=pix)
        doc.save(out_prefix + ".pdf"); return len(pages)


if __name__ == "__main__":
    import sys
    from .source_doc import parse
    src, out = sys.argv[1], sys.argv[2]; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    pg = Page(seed=seed); lines = parse(src); pages = pg.layout(lines)
    n = pg.render(pages, out)
    print(f"{n} page(s) -> {out}.pdf; missing glyphs: {dict(pg.missing.most_common(12))}")
