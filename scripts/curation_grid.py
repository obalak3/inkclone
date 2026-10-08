"""Labelled grids of every REAL pickable instance per glyph (variants inherit from these),
plus a JSON map grid label -> source key for blocklisting after visual review."""
import sys, pickle, json, numpy as np, pymupdf
sys.path.insert(0, "."); from inkclone.extract import render_strokes
L = pickle.load(open("data/derived/units.pkl", "rb"))["letters"]; S = pickle.load(open("data/derived/symbols_final.pkl", "rb"))
groups = {"lower": [(c, L.get(c, [])) for c in "abcdefghijklmnopqrstuvwxyz"],
          "upper": [(c, L.get(c, [])) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
          "digits": [(c, S.get(c, [])) for c in "0123456789"],
          "symbols": [(c, S.get(c, [])) for c in "()[]{}|=+-<>≤≥≠∞∈→←↔∪∩∅×/,.;:" if c in S]}
per, cell, xh = 24, 50, 8; labels = {}
for gname, rows in groups.items():
    rows = [(c, [u for u in insts if not u.get("synthetic")] or insts[:per]) for c, insts in rows]
    canvas = np.full((len(rows) * cell, (per + 1) * cell, 3), 255, np.uint8)
    page = pymupdf.open().new_page(width=canvas.shape[1] * 72 / 110, height=canvas.shape[0] * 72 / 110)
    for r, (c, insts) in enumerate(rows):
        for k, inst in enumerate(insts[:per]):
            img = render_strokes([p * [xh, xh, 1.0] + [6, 3.0 * xh, 0] for p in inst["pieces"]], (cell * 72 / 110,) * 2, dpi=110)
            h, w = min(img.shape[0], cell), min(img.shape[1], cell); canvas[r*cell:r*cell+h, (k+1)*cell:(k+1)*cell+w] = img[:h, :w]
            labels[f"{gname}:{c}{k+1}"] = list(inst["src"]) + [inst.get("k")]
    page.insert_image(page.rect, pixmap=pymupdf.Pixmap(pymupdf.csRGB, canvas.shape[1], canvas.shape[0], canvas.tobytes(), False))
    for r, (c, insts) in enumerate(rows):
        page.insert_text((3, (r * cell + 30) * 72 / 110), f"{c} {len(insts)}", fontsize=6, color=(0.8, 0, 0))
        for k in range(min(len(insts), per)): page.insert_text((((k + 1) * cell + 2) * 72 / 110, (r * cell + 48) * 72 / 110), str(k + 1), fontsize=4, color=(0.6, 0.6, 0.6))
    page.get_pixmap(dpi=130).save(f"reports/curation_{gname}.png")
json.dump(labels, open("data/curation/grid_labels.json", "w"))
print("grids written for", list(groups))
