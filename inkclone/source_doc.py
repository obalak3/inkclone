"""Parse a typed (LaTeX-produced) homework PDF into layout lines.

Each line: {"kind": header|code|prose|title|meta, "indent": int, "text": str}.
Headers come from bold fonts at section size; code lines from a block whose first
line looks like Name(args) and whose lines are indented in steps; the rest is prose.
Sub/superscripts are detected by smaller font size + baseline shift and marked
with _ and ^ (single char or {group}).
"""
import os
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
        size = max(sp["size"] for sp in num + den)
        def txt(L):
            out, base = "", max(L, key=lambda sp: sp["size"])["origin"][1]
            for sp in sorted(L, key=lambda sp: sp["bbox"][0]):
                t = sp["text"].strip()
                if sp["size"] < 0.8 * size and sp["origin"][1] < base - 0.1 * size: t = "^" + t if len(t) == 1 else "^{" + t + "}"
                elif sp["size"] < 0.8 * size and sp["origin"][1] > base + 0.1 * size: t = "_" + t if len(t) == 1 else "_{" + t + "}"
                out += t
            return out.replace(" ", "~")
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
        body_sz = sorted(sp["size"] for sp in spans)[len(spans) // 2] if spans else 10
        real = lambda sp: any(ord(c) >= 32 and not c.isspace() for c in sp["text"]) and not ("CMEX" in sp["font"] and sp["text"].strip() not in ("P", "X"))
        big = [g for g in groups if any(sp["size"] >= 0.8 * body_sz and real(sp) and len(sp["text"].strip()) >= 2 for sp in g["spans"])
               or sum(1 for sp in g["spans"] if sp["size"] >= 0.8 * body_sz and real(sp)) >= 3]
        rest = []
        for g in groups:
            if g in big: continue
            near = [b for b in big if abs(b["y"] - g["y"]) < 1.0 * body_sz]
            if near:
                # a subscript sits ~0.3 body sizes below its line, a superscript ~0.4 above: pick the line whose expected offset fits best
                def err(b):
                    dy = (g["y"] - b["y"]) / body_sz
                    return min(abs(dy - 0.3), abs(dy + 0.42))
                b = min(near, key=err); b["spans"].extend(g["spans"])
            else: rest.append(g)
        groups = sorted(big + rest, key=lambda g: g["y"])
        if os.environ.get("INK_DEBUG2"):
            for g in groups:
                if 360 < g["y"] < 400 and pno == 0: print("GROUP", round(g["y"], 1), "big" if g in big else "small", [(sp["text"], round(sp["size"], 1), round(sp["origin"][1], 1)) for sp in sorted(g["spans"], key=lambda sp: sp["bbox"][0])][:14])
        merged = []
        def attach_limits(prev, g):
            sps = g["spans"]
            if not prev or not all(sp["size"] < 0.8 * body_sz for sp in sps): return False
            ops = [sp for sp in prev["spans"] if sp["text"].strip().split("^")[0].split("_")[0] in ("min", "max", "Σ")]
            x0, x1 = min(sp["bbox"][0] for sp in sps), max(sp["bbox"][2] for sp in sps)
            for op in ops:
                if op["bbox"][2] > x0 - 25 and op["bbox"][0] < x1 + 25 and 0 < g["y"] - prev["y"] < 2.2 * body_sz:
                    lim = "".join(sp["text"] for sp in sorted(sps, key=lambda sp: sp["bbox"][0])).replace(" ", "").replace("ℓ", "l")
                    op["text"] = op["text"].strip() + "_{" + lim + "}"; return True
            return False
        def attach_upper(g, nxt):
            """a small group sitting just ABOVE a Σ/max/min in the next group -> superscript limit"""
            sps = g["spans"]
            if not all(sp["size"] < 0.8 * body_sz for sp in sps): return False
            x0, x1 = min(sp["bbox"][0] for sp in sps), max(sp["bbox"][2] for sp in sps)
            for op in nxt["spans"]:
                if op["text"].strip().split("_")[0] in ("Σ", "min", "max") and op["bbox"][2] > x0 - 25 and op["bbox"][0] < x1 + 25 and 0 < nxt["y"] - g["y"] < 2.2 * body_sz:
                    lim = "".join(sp["text"] for sp in sorted(sps, key=lambda sp: sp["bbox"][0])).replace(" ", "").replace("ℓ", "l")
                    op["text"] = op["text"].strip() + "^{" + lim + "}"; return True
            return False
        if os.environ.get("INK_DEBUG"):
            for g in groups:
                if 540 < g["y"] < 580: print("GROUP", round(g["y"], 1), [(sp["text"], round(sp["size"], 1)) for sp in g["spans"]])
        kept = []
        for i, g in enumerate(groups):
            if kept and attach_limits(kept[-1], g): continue
            if i + 1 < len(groups) and attach_upper(g, groups[i + 1]): continue
            # a big operator (Σ) on its own baseline belongs in the line it straddles: push it into the next group
            for sp in g["spans"]:
                if "CMEX" in sp["font"] and sp["text"].strip() in ("X", "P"): sp["text"] = "Σ"
            sig = [sp for sp in g["spans"] if sp["text"].strip().startswith("Σ")]
            empty_cmex = lambda sp: "CMEX" in sp["font"] and not any(ord(c) >= 32 and not c.isspace() for c in sp["text"])
            small = [sp for sp in g["spans"] if sp["size"] < 0.8 * body_sz]
            if sig and i + 1 < len(groups) and all(sp in sig or sp in small or empty_cmex(sp) for sp in g["spans"]) and 0 < groups[i + 1]["y"] - g["y"] < 1.5 * body_sz:
                if small and "^{" not in sig[0]["text"]:
                    sig[0]["text"] = sig[0]["text"].strip() + "^{" + "".join(sp["text"] for sp in sorted(small, key=lambda sp: sp["bbox"][0])).replace(" ", "").replace("ℓ", "l") + "}"
                groups[i + 1]["spans"].extend(sig); continue
            kept.append(g)
        groups = kept
        for g in groups:
            sps = sorted(g["spans"], key=lambda sp: sp["bbox"][0])
            bodyish = [sp for sp in sps if sp["size"] >= 0.8 * body_sz] or sps
            main = max(bodyish, key=lambda sp: len(sp["text"])); size = main["size"]; base = main["origin"][1]
            txt, prev_x1 = "", None
            run, run_mode = "", None
            def flush():
                nonlocal txt, run, run_mode
                if run:
                    r = run.replace(" ", "")
                    txt += (run_mode + r) if len(r) == 1 else (run_mode + "{" + r + "}")
                run, run_mode = "", None
            for sp in sps:
                t = sp["text"]
                if "CMEX" in sp["font"] and t.strip() in ("P", "X"): t = "Σ"
                if "CMEX" in sp["font"]:  # big delimiters: parens kept, braces and the rest dropped
                    t = "".join({"\x00": "(", "\x10": "(", "\x02": "(", "\x01": ")", "\x11": ")", "\x03": ")"}.get(ch, ch if ord(ch) >= 32 else "") for ch in t)
                if "CMMI" in sp["font"] and t.strip() == "" and sp["bbox"][2] - sp["bbox"][0] > 1.0: t = "l"
                t = t.replace("ℓ", "l")
                if "CMSY" in sp["font"] and t.strip() == "p": t = "√"
                small = sp["size"] < 0.8 * size
                mode = "_" if (small and sp["origin"][1] > base + 0.12 * size) else "^" if (small and sp["origin"][1] < base - 0.12 * size) else None
                if mode:
                    if run_mode and mode != run_mode: flush()
                    run_mode = mode; run += t.strip(); prev_x1 = sp["bbox"][2]; continue
                flush()
                if prev_x1 is not None and sp["bbox"][0] - prev_x1 > 0.18 * size: txt += " "
                txt += t; prev_x1 = sp["bbox"][2]
            flush()
            txt = txt.replace("̸=", "≠").replace("=⇒", "⇒").replace("⇐⇒", "⇔").replace("⇐=", "⇐").replace("· · ·", "...").replace(". . .", "...").replace("−", "-")
            txt = txt.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
            txt = re.sub(r"\biff(?=[a-z])", "iff ", txt)
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
            if re.match(r"^\(?[a-z]\)$", t):  # a bare sub-question label: attach to the next line
                lines.append({"kind": "prose", "indent": 0, "text": t, "bold_lead": True, "sublabel": True}); continue
            if is_short_bold and not re.match(r"^\([a-z]\)", t):
                n_headers += 1; in_code = False
                lines.append({"kind": "header", "indent": 0, "text": re.sub(r"^(\d+)\s+", r"\1) ", t)}); continue
            if re.match(r"^[A-Z][A-Za-z]+\(.*\)\s*$", t) and r["x"] <= body_x + 2:
                in_code = True; code_x0 = r["x"]; lines.append({"kind": "code", "indent": 0, "text": t}); continue
            if in_code and r["x"] > code_x0 + 2:
                lines.append({"kind": "code", "indent": max(1, round((r["x"] - code_x0) / 18)), "text": t}); continue
            in_code = False
            if re.match(r"^\d+$", t) or not re.search(r"[^\s()]", t): continue  # page number, or a line that held only big delimiters
            # a standalone equation line: no lowercase word of 3+ letters, indented or centred
            if not re.search(r"[a-z]{3,}", re.sub(r"(?<![A-Za-z])(min|max|log|lim|argmin|argmax)(?![A-Za-z])", "", t)) and r["x"] > body_x + 20:
                if os.environ.get("INK_DEBUG2"): print("MATH", pno + 1, round(r["y"], 1), repr(t[:40]), "x", round(r["x"]), "size", round(r["size"], 1))
                lines.append({"kind": "math", "indent": 2, "text": t}); continue
            lines.append({"kind": "prose", "indent": 0, "text": t, "bold_lead": r["bold"]})
    return merge_paragraphs(lines)


def merge_paragraphs(lines):
    """Join consecutive prose lines into paragraphs; a bold lead word (Why. / Time.) starts a new one.
    Anything before the first header is front matter (title, author, due date) and is kept as separate lines."""
    out = []; seen_header = False
    for ln in lines:
        if ln["kind"] == "header": seen_header = True
        starts_sub = ln["kind"] == "prose" and re.match(r"^\([a-z]\)", ln["text"]) is not None
        if out and out[-1].get("sublabel"):  # "(b)" alone on a line + the following text -> one paragraph,
            if starts_sub:                     # unless that text is itself sub-labelled: then "(b)" heads a group
                out[-1] = dict(out[-1], sublabel=False); out.append(dict(ln)); continue
            out[-1] = dict(out[-1], text=(out[-1]["text"] + " " + ln["text"]).strip(), sublabel=False, kind="prose"); continue
        if starts_sub: out.append(dict(ln)); continue
        if not seen_header and ln["kind"] == "prose" and len(ln["text"].split()) <= 8: out.append(dict(ln, kind="meta")); continue
        if ln["kind"] == "prose" and out and out[-1]["kind"] == "prose" and not ln.get("bold_lead"):
            out[-1]["text"] = (out[-1]["text"] + " " + ln["text"]).strip()
        else:
            out.append(dict(ln))
    return out


if __name__ == "__main__":
    import sys
    for ln in parse(sys.argv[1]): print(f"{ln['kind']:6s} {'  ' * ln['indent']}{ln['text']}")
