"""(1) real HW5 page next to the generated set05 page; (2) pick-the-fake sheet with sealed key."""
import sys, random, json, numpy as np, pymupdf
sys.path.insert(0, "."); from inkclone.page import Page, PAGE_W, PAGE_H; from inkclone.source_doc import parse; from inkclone.extract import render_strokes
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 11
pg = Page(seed=seed); pages = pg.layout(parse("data/sources/set05.pdf")); pg.render(pages, "reports/page_set05_eval", dpi=300)
# (1) side by side at equal scale
dpi = 150
real = pymupdf.open("data/raw/CS 3511 HW5.pdf")[0].get_pixmap(dpi=dpi); R = np.frombuffer(real.samples, np.uint8).reshape(real.h, real.w, real.n)[:, :, :3]
G = render_strokes(pages[0], (PAGE_W, PAGE_H), dpi=dpi)
H = max(R.shape[0], G.shape[0]); canvas = np.full((H + 30, R.shape[1] + G.shape[1] + 30, 3), 255, np.uint8)
canvas[30:30 + R.shape[0], :R.shape[1]] = R; canvas[30:30 + G.shape[0], R.shape[1] + 30:] = G
doc = pymupdf.open(); p = doc.new_page(width=canvas.shape[1] * 72 / dpi, height=canvas.shape[0] * 72 / dpi)
p.insert_image(p.rect, pixmap=pymupdf.Pixmap(pymupdf.csRGB, canvas.shape[1], canvas.shape[0], canvas.tobytes(), False))
p.insert_text((6, 10), "REAL: your HW5 page 1", fontsize=6, color=(0.7, 0, 0)); p.insert_text(((R.shape[1] + 36) * 72 / dpi, 10), "GENERATED from set05.pdf", fontsize=6, color=(0.7, 0, 0))
p.get_pixmap(dpi=200).save("reports/side_by_side_hw5.png")
# (2) pick the fake: words that exist as real whole-word instances AND can be assembled from units
rng = random.Random(seed); s = pg.s
cands = [t for t, insts in s.W.items() if t.isalpha() and 4 <= len(t) <= 9 and len(insts) >= 1]
rng.shuffle(cands); items = []
for t in cands:
    if len(items) >= 24: break
    real_u = rng.choice(s.W[t])
    saved = s.W; s.W = {}; s.lower = {}
    fake_strokes, fw, how = s.word(t, set()); s.W = saved; s.lower = {}
    for tt, insts in s.W.items(): s.lower.setdefault(tt.lower(), []).extend(insts)
    if how != "units" or not fake_strokes: continue
    rp, rw = s.conform(real_u["pieces"], real_u)
    items.append((t, "real", rp, rw)); items.append((t, "fake", fake_strokes, fw))
rng.shuffle(items)
xh, cell_w, cell_h, cols = 7.0, 150, 56, 4; rows = (len(items) + cols - 1) // cols
doc2 = pymupdf.open(); p2 = doc2.new_page(width=cols * cell_w, height=rows * cell_h + 20); key = []
for n, (t, kind, strokes, w) in enumerate(items):
    cx, cy = (n % cols) * cell_w, (n // cols) * cell_h + 20
    img = render_strokes([q * [xh, xh, 1.0] + [10, 2.8 * xh, 0] for q in strokes], (w * xh + 20, 4.6 * xh), dpi=300)
    pix = pymupdf.Pixmap(pymupdf.csRGB, img.shape[1], img.shape[0], np.ascontiguousarray(img).tobytes(), False)
    p2.insert_image(pymupdf.Rect(cx + 6, cy + 4, cx + 6 + img.shape[1] * 72 / 300, cy + 4 + img.shape[0] * 72 / 300), pixmap=pix)
    p2.insert_text((cx + 6, cy + cell_h - 4), f"#{n+1}", fontsize=7, color=(0.5, 0.5, 0.5)); key.append((n + 1, t, kind))
p2.insert_text((6, 12), "Which of these did you NOT write? (half are generated). Key in reports/pick_the_fake_KEY.txt", fontsize=7, color=(0.7, 0, 0))
p2.get_pixmap(dpi=200).save("reports/pick_the_fake.png")
open("reports/pick_the_fake_KEY.txt", "w").write("\n".join(f"#{n}: {t} -> {k}" for n, t, k in key) + "\n")
print(f"side-by-side + pick-the-fake written ({len(items)} items, {sum(1 for i in items if i[1]=='fake')} fake)")
