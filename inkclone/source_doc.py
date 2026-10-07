"""Parse a typed (LaTeX-produced) homework PDF into layout lines.

Each line: {"kind": header|code|prose|title|meta, "indent": int, "text": str}.
Headers come from bold fonts at section size; code lines from a block whose first
line looks like Name(args) and whose lines are indented in steps; the rest is prose.
Sub/superscripts are detected by smaller font size + baseline shift and marked
with _ and ^ (single char or {group}).
"""
import re
import pymupdf

GREEK = {"∞": "∞", "≤": "≤", "≥": "≥", "∈": "∈", "⇒": "⇒", "→": "→", "≠": "≠", "∅": "∅"}


def fold_fractions(page, spans):
    """LaTeX draws a fraction as a thin filled rule with numerator spans above and denominator spans
    below. Fold each such group into one synthetic span 'frac{num}{den}' sitting on the rule."""
    rules = []
    for dr in page.get_drawings():
        r = dr["rect"]
        if r.height < 1.2 and r.width > 3: rules.append(r)  # stroked or filled thin horizontal rule
    if not rules: return spans
    used = set(); out = []
    for r in rules:
        inside = lambda sp: sp["bbox"][0] >= r.x0 - 1.5 and sp["bbox"][2] <= r.x1 + 1.5   # span sits within the bar
        num = [sp for sp in spans if id(sp) not in used and inside(sp) and -0.5 < r.y0 - sp["bbox"][3] + 2 < 1.4 * sp["size"]]
        den = [sp for sp in spans if id(sp) not in used and inside(sp) and -0.5 < sp["bbox"][1] - r.y1 + 2 < 1.4 * sp["size"]]
        if not num or not den: continue
        txt = lambda L: "".join(sp["text"] for sp in sorted(L, key=lambda sp: sp["bbox"][0])).replace(" ", "~")
        size = max(sp["size"] for sp in num + den)
        synthetic = {"text": f"frac{{{txt(num)}}}{{{txt(den)}}}", "size": size, "font": num[0]["font"],
                     "origin": (r.x0, r.y1 + 0.3 * size), "bbox": (r.x0, min(sp["bbox"][1] for sp in num), r.x1, max(sp["bbox"][3] for sp in den))}
        for sp in num + den: used.add(id(sp))
        out.append(synthetic)
    return [sp for sp in spans if id(sp) not in used] + out


def parse(path):
    doc = pymupdf.open(path); lines = []
    for pno, page in enumerate(doc):
        d = page.get_text("dict"); spans = []
        for b in d["blocks"]:
            for l in b.get("lines", []):
                for sp in l["spans"]:
                    if sp["text"].strip(): spans.append(sp)
        spans = fold_fractions(page, spans)
        # group spans into baselines
        spans.sort(key=lambda sp: (sp["origin"][1], sp["bbox"][0]))
        groups = []
        for sp in spans:
            if groups and abs(sp["origin"][1] - groups[-1]["y"]) < 0.45 * max(sp["size"], groups[-1]["size"]) and sp["size"] >= 0.8 * groups[-1]["size"]:
                groups[-1]["spans"].append(sp)
            elif groups and abs(sp["origin"][1] - groups[-1]["y"]) < 0.6 * groups[-1]["size"]:
                groups[-1]["spans"].append(sp)  # sub/superscript on this line
            else:
                groups.append({"y": sp["origin"][1], "size": sp["size"], "spans": [sp]})
        merged = []
        for g in groups:
            sps = sorted(g["spans"], key=lambda sp: sp["bbox"][0])
            main = max(sps, key=lambda sp: len(sp["text"])); size = main["size"]; base = main["origin"][1]
            txt, prev_x1 = "", None
            for sp in sps:
                t = sp["text"]
                if "CMEX" in sp["font"] and t.strip() == "P": t = "Σ"
                if "CMSY" in sp["font"] and t.strip() == "p": t = "√"
                small = sp["size"] < 0.8 * size
                if small and sp["origin"][1] > base + 0.12 * size: t = ("_" + t) if len(t.strip()) == 1 else ("_{" + t.strip() + "}")
                elif small and sp["origin"][1] < base - 0.12 * size: t = ("^" + t) if len(t.strip()) == 1 else ("^{" + t.strip() + "}")
                if prev_x1 is not None and sp["bbox"][0] - prev_x1 > 0.18 * size and not t.startswith(("_", "^")): txt += " "
                txt += t; prev_x1 = sp["bbox"][2]
            txt = txt.replace("̸=", "≠").replace("=⇒", "⇒").replace("⇐⇒", "⇔").replace("· · ·", "...").replace(". . .", "...").replace("−", "-")
            txt = re.sub(r"\|\s*([A-Za-z])\s*\|", lambda m: "|" + m.group(1) + "|", txt)
            bold = "CMBX" in main["font"] or "Bold" in main["font"]
            merged.append({"x": sps[0]["bbox"][0], "y": g["y"], "size": size, "bold": bold, "text": re.sub(r"\s+", " ", txt).strip(), "font": main["font"]})
        body_x = min(r["x"] for r in merged if not r["bold"]) if merged else 0
        body_size = sorted(r["size"] for r in merged)[len(merged) // 2]
        in_code = False; code_x0 = None; n_headers = 0
        for i, r in enumerate(merged):
            t = re.sub(r"\s+", " ", r["text"]).strip()
            t = "".join(GREEK.get(c, c) for c in t)
            nwords = len(t.split())
            is_short_bold = r["bold"] and nwords <= 8 and r["size"] >= 0.95 * body_size
            if pno == 0 and i == 0 and r["size"] > 1.3 * body_size and r["bold"] and not re.match(r"^\d+\s", t):
                lines.append({"kind": "title", "indent": 0, "text": t}); continue
            if pno == 0 and n_headers == 0 and i < 4 and nwords <= 8 and not is_short_bold and len(t) < 60:
                lines.append({"kind": "meta", "indent": 0, "text": t}); continue
            if is_short_bold:
                n_headers += 1; in_code = False
                lines.append({"kind": "header", "indent": 0, "text": re.sub(r"^(\d+)\s+", r"\1) ", t)}); continue
            if re.match(r"^[A-Z][A-Za-z]+\(.*\)\s*$", t) and r["x"] <= body_x + 2:
                in_code = True; code_x0 = r["x"]; lines.append({"kind": "code", "indent": 0, "text": t}); continue
            if in_code and r["x"] > code_x0 + 2:
                lines.append({"kind": "code", "indent": max(1, round((r["x"] - code_x0) / 18)), "text": t}); continue
            in_code = False
            if re.match(r"^\d+$", t): continue  # page number
            # a standalone equation line: no lowercase word of 3+ letters, indented or centred
            if not re.search(r"[a-z]{3,}", t) and r["x"] > body_x + 20:
                lines.append({"kind": "math", "indent": 2, "text": t}); continue
            lines.append({"kind": "prose", "indent": 0, "text": t, "bold_lead": r["bold"]})
    return merge_paragraphs(lines)


def merge_paragraphs(lines):
    """Join consecutive prose lines into paragraphs; a bold lead word (Why. / Time.) starts a new one.
    Anything before the first header is front matter (title, author, due date) and is kept as separate lines."""
    out = []; seen_header = False
    for ln in lines:
        if ln["kind"] == "header": seen_header = True
        if not seen_header and ln["kind"] == "prose" and len(ln["text"].split()) <= 8: out.append(dict(ln, kind="meta")); continue
        if ln["kind"] == "prose" and out and out[-1]["kind"] == "prose" and not ln.get("bold_lead"):
            out[-1]["text"] = (out[-1]["text"] + " " + ln["text"]).strip()
        else:
            out.append(dict(ln))
    return out


if __name__ == "__main__":
    import sys
    for ln in parse(sys.argv[1]): print(f"{ln['kind']:6s} {'  ' * ln['indent']}{ln['text']}")
