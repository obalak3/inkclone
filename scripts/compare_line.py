"""Original handwritten line (page 2 of the calc notebook) above several synthesised versions of the same text, plus an unseen sentence."""
import sys, numpy as np, pymupdf
sys.path.insert(0, "."); from inkclone.extract import render_strokes; from inkclone.synth import Synth
doc = pymupdf.open("data/raw/Multivariable Calculus.pdf"); pg = doc[1]
dpi = 600; S = dpi / 72
orig = pg.get_pixmap(dpi=dpi, clip=pymupdf.Rect(12, 12, 310, 50), colorspace=pymupdf.csRGB)
orig = np.frombuffer(orig.samples, np.uint8).reshape(orig.h, orig.w, 3)
rows = [("ORIGINAL (your page 2)", orig)]
for seed, text in [(1, "Distance between two vectors"), (2, "Distance between two vectors"), (3, "Distance between two vectors"),
                   (4, "the quick brown fox jumps over the lazy dog"), (5, "shortest paths have at most k edges so stop after round k")]:
    s = Synth(seed=seed); strokes, width = s.line(text, xh_pt=4.0)
    xh = 4.0; img = render_strokes([p + [6, 2.6 * xh, 0] for p in strokes], (width + 12, 4.6 * xh), dpi=dpi)
    rows.append((f"SYNTH seed {seed}: {text}", img))
W = max(r[1].shape[1] for r in rows) + 20; H = sum(r[1].shape[0] + 40 for r in rows)
canvas = np.full((H, W, 3), 255, np.uint8); y = 0
page = pymupdf.open().new_page(width=W * 72 / dpi, height=H * 72 / dpi)
for label, img in rows:
    canvas[y + 30:y + 30 + img.shape[0], 10:10 + img.shape[1]] = img; y += img.shape[0] + 40
pix = pymupdf.Pixmap(pymupdf.csRGB, W, H, canvas.tobytes(), False); page.insert_image(page.rect, pixmap=pix)
y = 0
for label, img in rows:
    page.insert_text(((10) * 72 / dpi, (y + 20) * 72 / dpi), label, fontsize=5, color=(0.7, 0, 0)); y += img.shape[0] + 40
page.get_pixmap(dpi=300).save("reports/compare_line_v2.png"); print("wrote reports/compare_line_v2.png")
