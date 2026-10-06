"""Re-render page 2 from recovered strokes and compare with the PDF's own rendering."""
import sys, numpy as np, pymupdf
sys.path.insert(0, "."); from inkclone.extract import extract_page, render_strokes
doc = pymupdf.open("data/raw/Multivariable Calculus.pdf"); pg = doc[1]
pdata = extract_page(pg)
print(f"page 2: {len(pdata['strokes'])} strokes, samples/stroke median {int(np.median([len(s) for s in pdata['strokes']]))}")
def iou(dpi, clip=None):
    ours = render_strokes(pdata["strokes"], pdata["size"], dpi=dpi)
    pix = pg.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    ref = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w)
    H, W = min(ref.shape[0], ours.shape[0]), min(ref.shape[1], ours.shape[1])
    a = ref[:H, :W] < 128; b = ours[:H, :W, 0] < 128
    if clip:
        S = dpi/72; sl = (slice(int(clip[1]*S), int(clip[3]*S)), slice(int(clip[0]*S), int(clip[2]*S))); a, b = a[sl], b[sl]
    i, u = (a & b).sum(), (a | b).sum()
    return i/u, i/a.sum(), i/b.sum(), ref, ours, H, W
for dpi, clip in [(150, None), (600, (14, 13, 300, 48))]:
    r = iou(dpi, clip); print(f"dpi {dpi} clip {clip}: IoU {r[0]:.3f} recall {r[1]:.3f} precision {r[2]:.3f}")
_, _, _, ref, ours, H, W = iou(150)
side = np.full((H, W*2+10, 3), 255, np.uint8)
side[:, :W] = np.repeat(ref[:H, :W, None], 3, 2); side[:, W+10:] = ours[:H, :W]
pymupdf.Pixmap(pymupdf.csRGB, side.shape[1], side.shape[0], side.tobytes(), False).save("reports/validate_p2_side_by_side.png")
S = 600/72; clip = (14, 13, 99, 48); z = render_strokes(pdata["strokes"], pdata["size"], dpi=600)
y0, y1, x0, x1 = int(clip[1]*S), int(clip[3]*S), int(clip[0]*S), int(clip[2]*S)
pymupdf.Pixmap(pymupdf.csRGB, x1-x0, y1-y0, np.ascontiguousarray(z[y0:y1, x0:x1]).tobytes(), False).save("reports/validate_distance_600dpi.png")
