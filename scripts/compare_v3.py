"""(a) filtered-letter grid for a e t s; (b) set05 sentences synthesised with Synth2 next to the real HW5 line."""
import sys, pickle, numpy as np, pymupdf
sys.path.insert(0, "."); from inkclone.extract import render_strokes; from inkclone.synth2 import Synth2
L = pickle.load(open("data/derived/units.pkl", "rb"))["letters"]
letters, per, cell, xh = "aets", 30, 60, 12
canvas = np.full((len(letters) * cell, per * cell, 3), 255, np.uint8)
for r, c in enumerate(letters):
    for k, inst in enumerate(L[c][:per]):
        img = render_strokes([p * [xh, xh, 1.0] + [6, 2.2 * xh, 0] for p in inst["pieces"]], (cell * 72 / 110,) * 2, dpi=110)
        h, w = min(img.shape[0], cell), min(img.shape[1], cell); canvas[r*cell:r*cell+h, k*cell:k*cell+w] = img[:h, :w]
pymupdf.Pixmap(pymupdf.csRGB, canvas.shape[1], canvas.shape[0], canvas.tobytes(), False).save("reports/units_letter_grid_aets.png")

dpi = 600
pg = pymupdf.open("data/raw/CS 3511 HW5.pdf")[0]
orig = pg.get_pixmap(dpi=dpi, clip=pymupdf.Rect(8, 5, 300, 40), colorspace=pymupdf.csRGB)
rows = [("ORIGINAL HW5 line", np.frombuffer(orig.samples, np.uint8).reshape(orig.h, orig.w, 3))]
texts = ["Shortest Paths with Few Steps", "Bellman Ford, stop when a round changes nothing",
         "after round i, distance is the best path with at most i edges",
         "No change round means fixed point, so stopping early is safe", "the two vectors are parallel if the cross product is zero"]
for seed, text in enumerate(texts, 1):
    s = Synth2(seed=seed); strokes, width, how = s.line(text, xh_pt=4.2)
    img = render_strokes([p + [6, 2.6 * 4.2, 0] for p in strokes], (width + 12, 4.6 * 4.2), dpi=dpi)
    rows.append((f"SYNTH2: {text}   [{' '.join('W' if h == 'word' else 'u' for h in how)}]", img))
W = max(r[1].shape[1] for r in rows) + 20; H = sum(r[1].shape[0] + 40 for r in rows)
canvas = np.full((H, W, 3), 255, np.uint8); y = 0
for _, img in rows: canvas[y+30:y+30+img.shape[0], 10:10+img.shape[1]] = img; y += img.shape[0] + 40
page = pymupdf.open().new_page(width=W * 72 / dpi, height=H * 72 / dpi)
page.insert_image(page.rect, pixmap=pymupdf.Pixmap(pymupdf.csRGB, W, H, canvas.tobytes(), False)); y = 0
for label, img in rows: page.insert_text((10 * 72 / dpi, (y + 20) * 72 / dpi), label, fontsize=4.5, color=(0.7, 0, 0)); y += img.shape[0] + 40
page.get_pixmap(dpi=300).save("reports/compare_line_v5.png"); print("wrote both")
