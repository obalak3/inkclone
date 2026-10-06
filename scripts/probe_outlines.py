import pymupdf, math, collections, numpy as np
doc = pymupdf.open("data/raw/Multivariable Calculus.pdf")
pg = doc[1]; d = pg.get_drawings()
def subpaths(p):
    subs, cur = [], []
    for it in p["items"]:
        a, b = it[1], it[-1]
        if cur and (abs(cur[-1].x-a.x) > 1e-3 or abs(cur[-1].y-a.y) > 1e-3):
            subs.append(cur); cur = []
        if not cur: cur.append(a)
        cur.append(b)
    if cur: subs.append(cur)
    return subs
counts = collections.Counter(len(subpaths(p)) for p in d)
print("subpaths per drawing:", counts.most_common(6))
# are subpaths closed loops? estimate width of each loop via loop area / half-perimeter
def loop_stats(pts):
    n=len(pts); closed = math.dist(pts[0], pts[-1]) < 0.5
    area = abs(sum(pts[i].x*pts[(i+1)%n].y - pts[(i+1)%n].x*pts[i].y for i in range(n)))/2
    per = sum(math.dist(pts[i], pts[(i+1)%n]) for i in range(n))
    return closed, area, per, 2*area/per  # ~ mean stroke width for a thin ribbon
widths=[]; closedc=collections.Counter()
for p in d:
    for s in subpaths(p):
        c,a,per,w = loop_stats(s); closedc[c]+=1; widths.append(w)
print("loops closed?", closedc, "| est. stroke width pt: median %.2f  p10 %.2f  p90 %.2f" % tuple(np.percentile(widths,[50,10,90])))
# skeleton test on the word 'Distance' (first OCR box) via rasterize + skimage
from skimage.morphology import skeletonize
from skimage.measure import label
clip = pymupdf.Rect(14, 13, 99, 48); dpi = 600
pix = pg.get_pixmap(dpi=dpi, clip=clip, colorspace=pymupdf.csGRAY)
img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w) < 128
sk = skeletonize(img)
print("word 'Distance': ink px", img.sum(), "skeleton px", sk.sum(), "components", label(sk, connectivity=2).max())
out = np.full((pix.h, pix.w, 3), 255, np.uint8); out[img] = (200,200,200); out[sk] = (200,0,0)
pymupdf.Pixmap(pymupdf.csRGB, pix.w, pix.h, out.tobytes(), False).save("skel.png")
