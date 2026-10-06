import pymupdf
doc = pymupdf.open("data/raw/Multivariable Calculus.pdf")
pg = doc[1]
print(repr(pg.get_text()[:700]))
print("--- words with boxes (first 12):")
for w in pg.get_text("words")[:12]: print([round(v,1) for v in w[:4]], w[4])
# render mode: is text invisible?
for b in pg.get_text("dict")["blocks"][:1]:
    for l in b.get("lines", [])[:1]:
        for s in l["spans"][:1]: print("span flags/font/size/color:", s["flags"], s["font"], s["size"], s["color"])
raw = pg.read_contents()
print("Tr (text render mode) ops present:", b"3 Tr" in raw, "| content bytes:", len(raw))
print("--- per-page summary:")
for i, p in enumerate(doc):
    d = p.get_drawings()
    print(f"p{i+1:2d} drawings={len(d):5d} images={len(p.get_images()):2d} textchars={len(p.get_text()):5d}")
