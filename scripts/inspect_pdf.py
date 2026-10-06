import pymupdf, collections, sys
doc = pymupdf.open("data/raw/Multivariable Calculus.pdf")
print("pages:", len(doc))
for i in [1]:  # page 2
    pg = doc[i]
    print("page", i+1, "size", pg.rect, "images:", len(pg.get_images()), "text chars:", len(pg.get_text()))
    d = pg.get_drawings()
    print("drawings:", len(d))
    types = collections.Counter(x["type"] for x in d)
    print("types (s=stroke,f=fill,fs=both):", types)
    widths = collections.Counter(round(x.get("width") or 0, 2) for x in d)
    print("stroke widths:", widths.most_common(8))
    itemkinds = collections.Counter(it[0] for x in d for it in x["items"])
    print("segment kinds (l=line,c=curve,re=rect,qu=quad):", itemkinds)
    npts = [len(x["items"]) for x in d]
    print("items per path: min/median/max", min(npts), sorted(npts)[len(npts)//2], max(npts))
    colors = collections.Counter(tuple(round(c,2) for c in (x.get("color") or x.get("fill") or ())) for x in d)
    print("colors:", colors.most_common(5))
    for x in d[:3]:
        print("---", x["type"], "w", x.get("width"), "rect", x["rect"], "n", len(x["items"]))
        for it in x["items"][:4]: print("   ", it)
pg = doc[1]
pix = pg.get_pixmap(dpi=110); pix.save("p2.png")
