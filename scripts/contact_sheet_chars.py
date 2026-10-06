"""Grid of segmented words: letters alternate colours, label under each word."""
import sys, pickle, random, numpy as np, pymupdf
sys.path.insert(0, "."); from inkclone.extract import render_strokes; from inkclone.segment import load_words
out = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
N_PICK = int(sys.argv[3]) if len(sys.argv) > 3 else 40; cols = int(sys.argv[4]) if len(sys.argv) > 4 else 5
d = pickle.load(open("data/derived/segments.pkl", "rb")); segs = d["segs"]
words = load_words(sorted({w["source"] for w in d["words"]}))
idx = [i for i, s in enumerate(segs) if s and len(words[i]["text"]) >= 3]
random.seed(seed); pick = random.sample(idx, N_PICK)
cw, ch, dpi = 1500 // cols, 95 * 5 // cols, 200; rows = (len(pick) + cols - 1) // cols
doc = pymupdf.open(); page = doc.new_page(width=cols*cw, height=rows*ch)
COLORS = [(200, 30, 30), (30, 90, 200), (20, 150, 60), (190, 120, 0)]
for n, k in enumerate(pick):
    w = words[k]; x0, y0, x1, y1 = w["bbox"]; sc = min((cw-20)/(x1-x0), (ch-35)/(y1-y0), 3.0)
    size = ((x1-x0)*sc, (y1-y0)*sc); S = dpi/72
    img = np.full((int(size[1]*S)+1, int(size[0]*S)+1, 3), 255, np.uint8)
    for c, chd in enumerate(segs[k]):
        pieces = [(w["strokes"][i][i0:i1] - [x0, y0, 0]) * [sc, sc, sc] for i, i0, i1 in chd["pieces"]]
        layer = render_strokes(pieces, size, dpi=dpi, color=COLORS[c % len(COLORS)])
        m = layer[:, :, 0] < 250; img[m] = layer[m]
    pix = pymupdf.Pixmap(pymupdf.csRGB, img.shape[1], img.shape[0], np.ascontiguousarray(img).tobytes(), False)
    cx, cy = (n % cols) * cw, (n // cols) * ch
    page.insert_image(pymupdf.Rect(cx+10, cy+5, cx+10+size[0], cy+5+size[1]), pixmap=pix)
    joins = "".join("_" if chd["joined_next"] else " " for chd in segs[k][:-1])
    page.insert_text((cx+10, cy+ch-8), f"{w['text']}   joins: {joins}", fontsize=9, color=(0.3, 0.3, 0.3))
page.get_pixmap(dpi=110).save(out); print("wrote", out)
