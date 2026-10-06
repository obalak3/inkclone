import pymupdf
doc = pymupdf.open("data/raw/Multivariable Calculus.pdf")
pg = doc[1]
pg.get_pixmap(dpi=120).save("p2.png")
# zoom on the first word "Distance" and overlay the OCR word boxes
clip = pymupdf.Rect(10, 10, 310, 50)
pix = pg.get_pixmap(dpi=500, clip=clip); pix.save("p2_word.png")
# outline structure of the first path: is it a closed loop? sample path points
d = pg.get_drawings()
p = d[0]
pts = [it[1] for it in p["items"]] + [p["items"][-1][-1]]
print("closed:", p.get("closePath"), "first", pts[0], "last", pts[-1], "n", len(pts))
# width estimate: distance between point i and the point halfway around the loop
import math
n = len(pts)
ws = [math.dist(pts[i], pts[(i + n//2) % n]) for i in range(0, n, max(1, n//10))]
print("outline-to-opposite-side distances (pt):", [round(w,2) for w in ws])
