"""Extract pen strokes (centerline + width) from a GoodNotes PDF export.

GoodNotes writes each pen stroke as one filled PDF path made of many small closed
"ribbon" loops laid down in pen order. A loop's outline runs out along one side of
the ink and back along the other, so pairing opposite outline points recovers the
pen centerline and the ink width (pressure proxy) for that piece. Chaining the
pieces in loop order gives the whole stroke, oriented in drawing direction.
"""
import math, json
import numpy as np
import pymupdf


def flatten_loops(items, n=6):
    """PDF drawing items -> list of closed loops as (k,2) arrays, beziers flattened."""
    loops, cur = [], []
    for it in items:
        a = (it[1].x, it[1].y)
        if cur and math.dist(cur[-1], a) > 1e-3:
            loops.append(np.array(cur)); cur = []
        if not cur:
            cur.append(a)
        if it[0] == 'c':
            p = np.array([(q.x, q.y) for q in it[1:5]])
            t = np.linspace(0, 1, n + 1)[1:, None]
            pts = (1-t)**3*p[0] + 3*(1-t)**2*t*p[1] + 3*(1-t)*t**2*p[2] + t**3*p[3]
            cur.extend(map(tuple, pts))
        else:
            cur.append((it[-1].x, it[-1].y))
    if cur:
        loops.append(np.array(cur))
    return loops


def loop_centerline(L):
    """Closed ribbon outline (k,2) -> (m,3) array of x, y, width along the piece.

    Tries every rotation offset of the outline and keeps the pairing of opposite
    points whose summed pair distance is smallest (the across-the-ink pairing).
    """
    if math.dist(L[0], L[-1]) < 1e-6:
        L = L[:-1]
    n = len(L)
    if n < 4:
        c = L.mean(0); return np.array([[c[0], c[1], 0.0]])
    half = n // 2
    # all rotations at once: R[k] is the outline starting at point k
    idx = (np.arange(n)[None, :] + np.arange(n)[:, None]) % n
    R = L[idx]
    A, B = R[:, :half], R[:, half:2*half][:, ::-1]
    k = np.linalg.norm(A - B, axis=2).sum(1).argmin()
    A, B = A[k], B[k]
    mid = (A + B) / 2
    w = np.linalg.norm(A - B, axis=1)
    return np.column_stack([mid, w])


def stroke_from_path(items):
    """One PDF path (one pen stroke) -> (m,3) centerline in pen order."""
    pieces = [loop_centerline(L) for L in flatten_loops(items)]
    out = []
    for i, P in enumerate(pieces):
        if out:
            prev_end = out[-1][-1, :2]
            if math.dist(P[-1, :2], prev_end) < math.dist(P[0, :2], prev_end):
                P = P[::-1]
        elif len(pieces) > 1:
            nxt = pieces[1][:, :2].mean(0)
            if math.dist(P[0, :2], nxt) < math.dist(P[-1, :2], nxt):
                P = P[::-1]
        out.append(P)
    S = np.vstack(out)
    # drop near-duplicate consecutive samples where pieces overlap
    keep = np.r_[True, np.linalg.norm(np.diff(S[:, :2], axis=0), axis=1) > 0.05]
    S = S[keep]
    # pairing noise shows up as width jitter; a short median filter removes it
    if len(S) >= 5:
        from scipy.ndimage import median_filter
        S[:, 2] = median_filter(S[:, 2], size=5, mode='nearest')
    return S


def extract_page(page):
    strokes = [stroke_from_path(d["items"]) for d in page.get_drawings() if d["type"] == "f"]
    words = [{"bbox": [round(v, 2) for v in w[:4]], "text": w[4]} for w in page.get_text("words")]
    return {"size": [page.rect.width, page.rect.height], "strokes": strokes, "words": words}


def extract_pdf(path):
    doc = pymupdf.open(path)
    return [extract_page(p) for p in doc]


def save(pages, out_prefix):
    """pages -> <prefix>.npz (strokes) + <prefix>.json (words, sizes, stroke index)."""
    arrays, meta = {}, []
    for pi, pg in enumerate(pages):
        for si, s in enumerate(pg["strokes"]):
            arrays[f"p{pi}_s{si}"] = s.astype(np.float32)
        meta.append({"size": pg["size"], "n_strokes": len(pg["strokes"]), "words": pg["words"]})
    np.savez_compressed(out_prefix + ".npz", **arrays)
    json.dump(meta, open(out_prefix + ".json", "w"))


def render_strokes(strokes, size, dpi=150, color=(0, 0, 0)):
    """Draw strokes as variable-width round-capped polylines -> (H,W,3) uint8."""
    from skimage.draw import disk, polygon
    S = dpi / 72
    H, W = int(size[1] * S) + 1, int(size[0] * S) + 1
    img = np.full((H, W, 3), 255, np.uint8)
    for s in strokes:
        P = s[:, :2] * S; w = np.maximum(s[:, 2] * S, 0.8)
        for i in range(len(P)):
            rr, cc = disk((P[i, 1], P[i, 0]), w[i] / 2, shape=(H, W)); img[rr, cc] = color
            if i:
                d = P[i] - P[i-1]; nrm = np.hypot(*d)
                if nrm < 1e-6: continue
                nx, ny = -d[1] / nrm, d[0] / nrm
                a, b = w[i-1] / 2, w[i] / 2
                quad = np.array([P[i-1] + (nx*a, ny*a), P[i] + (nx*b, ny*b), P[i] - (nx*b, ny*b), P[i-1] - (nx*a, ny*a)])
                rr, cc = polygon(quad[:, 1], quad[:, 0], shape=(H, W)); img[rr, cc] = color
    return img


if __name__ == "__main__":
    import sys, os
    src = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else "data/derived/" + os.path.splitext(os.path.basename(src))[0]
    pages = extract_pdf(src); save(pages, out)
    n = sum(len(p["strokes"]) for p in pages); pts = sum(len(s) for p in pages for s in p["strokes"])
    print(f"{len(pages)} pages, {n} strokes, {pts} samples -> {out}.npz/.json")
