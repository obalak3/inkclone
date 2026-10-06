"""How does GoodNotes split one pen stroke into closed sub-loops? Are they sequential along the stroke?"""
import pymupdf, math, numpy as np, collections
from skimage.draw import polygon as draw_polygon
doc = pymupdf.open("data/raw/Multivariable Calculus.pdf")
pg = doc[1]; d = pg.get_drawings()

def flatten(items, n=6):
    """items -> list of closed loops as point arrays (bezier flattened)."""
    loops, cur = [], []
    for it in items:
        a, b = it[1], it[-1]
        if cur and math.dist(cur[-1], (a.x, a.y)) > 1e-3:
            loops.append(np.array(cur)); cur = []
        if not cur: cur.append((a.x, a.y))
        if it[0] == 'c':
            p0, p1, p2, p3 = [np.array((p.x, p.y)) for p in it[1:5]]
            for t in np.linspace(0, 1, n + 1)[1:]:
                q = (1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3
                cur.append(tuple(q))
        else:
            cur.append((b.x, b.y))
    if cur: loops.append(np.array(cur))
    return loops

for idx in [0, 5, 20]:
    loops = flatten(d[idx]["items"])
    print(f"path {idx}: {len(loops)} loops, rect {d[idx]['rect']}")
    prev = None
    for k, L in enumerate(loops):
        c = L.mean(0); bb = (L.min(0), L.max(0))
        gap = math.dist(c, prev) if prev is not None else float('nan')
        print(f"  loop {k}: pts={len(L):3d} center=({c[0]:.1f},{c[1]:.1f}) size=({bb[1][0]-bb[0][0]:.1f}x{bb[1][1]-bb[0][1]:.1f}) dist-from-prev-center={gap:.1f}")
        prev = c

# visualize path 0 loops color-coded, high zoom
idx = 0; loops = flatten(d[idx]["items"]); r = d[idx]["rect"]; S = 40
W, H = int((r.x1-r.x0)*S)+20, int((r.y1-r.y0)*S)+20
img = np.full((H, W, 3), 255, np.uint8)
cols = [(220,40,40),(40,160,40),(40,80,220),(200,140,0),(150,0,180),(0,160,160),(120,120,120),(230,100,160),(90,60,0),(0,0,0)]
for k, L in enumerate(loops):
    rr, cc = draw_polygon((L[:,1]-r.y0)*S+10, (L[:,0]-r.x0)*S+10, img.shape[:2])
    img[rr, cc] = cols[k % len(cols)]
pymupdf.Pixmap(pymupdf.csRGB, W, H, img.tobytes(), False).save("reports/probe_loops_path0.png")

# dataset sizing across the whole notebook
tot_paths = tot_loops = 0; words = []
for p in doc:
    dd = p.get_drawings(); tot_paths += len(dd)
    tot_loops += sum(len(flatten(x["items"], n=1)) for x in dd)
    words += [w[4] for w in p.get_text("words")]
alpha = [w for w in words if w.isalpha() and len(w) >= 2]
print(f"\nnotebook: {tot_paths} strokes, {tot_loops} loops, {len(words)} OCR tokens, {len(alpha)} alphabetic words (len>=2), {len(set(a.lower() for a in alpha))} distinct")
print("sample words:", collections.Counter(a.lower() for a in alpha).most_common(25))
