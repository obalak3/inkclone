"""Render a grid of labeled word samples (ink above, OCR label below) for eyeballing."""
import sys, json, random, numpy as np, pymupdf
sys.path.insert(0, "."); from inkclone.extract import render_strokes
prefix = sys.argv[1]; out = sys.argv[2]; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
arrs = np.load(prefix + ".npz"); meta = json.load(open(prefix + ".json"))
random.seed(seed); pick = random.sample(range(len(meta)), min(48, len(meta)))
cols, cw, ch, dpi = 6, 260, 90, 200
rows = (len(pick) + cols - 1) // cols
doc = pymupdf.open(); page = doc.new_page(width=cols*cw, height=rows*ch)
for n, k in enumerate(pick):
    m = meta[k]; strokes = [arrs[f"w{k}_s{j}"] for j in range(m["n_strokes"])]
    x0, y0, x1, y1 = m["bbox"]; sc = min((cw-20)/(x1-x0), (ch-35)/(y1-y0), 3.0)
    img = render_strokes([(s - [x0, y0, 0]) * [sc, sc, sc] for s in strokes], ((x1-x0)*sc, (y1-y0)*sc), dpi=dpi)
    pix = pymupdf.Pixmap(pymupdf.csRGB, img.shape[1], img.shape[0], np.ascontiguousarray(img).tobytes(), False)
    cx, cy = (n % cols) * cw, (n // cols) * ch
    page.insert_image(pymupdf.Rect(cx+10, cy+5, cx+10+(x1-x0)*sc, cy+5+(y1-y0)*sc), pixmap=pix)
    page.insert_text((cx+10, cy+ch-8), f"{m['text']}  (p{m['page']+1}, {m['n_strokes']} strokes)", fontsize=9, color=(0.8, 0, 0))
page.get_pixmap(dpi=110).save(out); print("wrote", out, "with", len(pick), "samples")
