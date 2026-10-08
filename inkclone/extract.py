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


def loop_centerline(L, n_out=None):
    """Closed ribbon outline (k,2) -> (m,3) array of x, y, width along the piece.

    A piece is a thin capsule: find its two end caps (extreme points along the principal
    axis), split the outline into the two long sides between them, and pair the sides at
    equal fractions of their arc length. The midpoints are the pen path, the pair distance
    the ink width."""
    if math.dist(L[0], L[-1]) < 1e-6:
        L = L[:-1]
    n = len(L)
    if n < 6:
        c = L.mean(0); return np.array([[c[0], c[1], 0.0]])
    c = L.mean(0); X = L - c
    # principal axis
    cov = X.T @ X; evals, evecs = np.linalg.eigh(cov); axis = evecs[:, np.argmax(evals)]
    proj = X @ axis; i1, i2 = int(np.argmin(proj)), int(np.argmax(proj))
    if i1 == i2: cc = L.mean(0); return np.array([[cc[0], cc[1], 0.0]])
    # the two sides, each from cap i1 to cap i2 (one forwards around the loop, one backwards)
    idx_f = [(i1 + k) % n for k in range((i2 - i1) % n + 1)]
    idx_b = [(i1 - k) % n for k in range((i1 - i2) % n + 1)]
    A, B = L[idx_f], L[idx_b]
    def by_fraction(P, m):
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        if d[-1] < 1e-9: return np.repeat(P[:1], m, 0)
        t = np.linspace(0, d[-1], m); return np.column_stack([np.interp(t, d, P[:, 0]), np.interp(t, d, P[:, 1])])
    m = n_out or max(4, n // 2)
    Af, Bf = by_fraction(A, m), by_fraction(B, m)
    mid = (Af + Bf) / 2; wpair = np.linalg.norm(Af - Bf, axis=1)
    # width: exact ribbon geometry (area = w*len, perimeter = 2len + 2w), robust against cap effects
    x, y = L[:, 0], L[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    per = np.linalg.norm(np.diff(np.vstack([L, L[:1]]), axis=0), axis=1).sum()
    half = per / 2; disc = half * half - 4 * area
    wv = (half - math.sqrt(disc)) / 2 if disc > 0 else (2 * area / per if per > 0 else 0.0)
    # trim the cap regions where the sides converge (pair distance much smaller than the width)
    keep = wpair > 0.35 * wv if wv > 0 else np.ones(m, bool)
    if keep.sum() >= 2: mid = mid[keep]
    return np.column_stack([mid, np.full(len(mid), wv)])


def _smooth_resample(S, step=0.1, window_pt=0.5):
    """Resample a chained centerline by arc length and smooth x/y within ~one pen width.
    The true pen path under GoodNotes' ribbons is smooth; this removes the piece-joint kinks."""
    P, w = S[:, :2], S[:, 2]
    d = np.linalg.norm(np.diff(P, axis=0), axis=1); cum = np.r_[0, np.cumsum(d)]
    if cum[-1] < 2 * step or len(P) < 4:
        return S
    t = np.r_[np.arange(0, cum[-1], step), cum[-1]]
    Q = np.column_stack([np.interp(t, cum, P[:, 0]), np.interp(t, cum, P[:, 1])]); wq = np.interp(t, cum, w)
    k = max(3, int(round(window_pt / step)) | 1); pad = k // 2
    if len(Q) > k:
        ker = np.ones(k) / k
        xs = np.convolve(np.r_[np.full(pad, Q[0, 0]), Q[:, 0], np.full(pad, Q[-1, 0])], ker, mode="valid")
        ys = np.convolve(np.r_[np.full(pad, Q[0, 1]), Q[:, 1], np.full(pad, Q[-1, 1])], ker, mode="valid")
        Q = np.column_stack([xs, ys]); Q[0], Q[-1] = P[0], P[-1]
    return np.column_stack([Q, wq])


def stroke_from_path(items, overlap_max=1.0):
    """One PDF path (one pen stroke) -> (m,3) centerline in pen order.

    Consecutive ribbon pieces overlap, so each new piece is trimmed to start where it passes
    closest to the end of the chain so far (never doubling back), then the chain is resampled
    and smoothed within one pen width."""
    loops = flatten_loops(items)
    pieces = [loop_centerline(L) for L in loops]
    cents = np.array([L.mean(0) for L in loops])
    out = []
    for i, P in enumerate(pieces):
        # orient each piece along the direction of travel given by neighbouring loop centroids
        # (loop centroids advance monotonically along the pen path; piece ends alone are ambiguous)
        lo, hi = max(0, i - 1), min(len(cents) - 1, i + 1)
        v = cents[hi] - cents[lo]
        if np.linalg.norm(v) > 1e-6 and len(P) > 1 and np.dot(P[-1, :2] - P[0, :2], v) < 0:
            P = P[::-1]
        if out:
            prev_end = out[-1][-1, :2]
            dist = np.linalg.norm(P[:, :2] - prev_end, axis=1); j = int(np.argmin(dist))
            if dist[j] < overlap_max:
                P = P[j + 1:]
        if len(P):
            out.append(P)
    if not out:
        return np.zeros((0, 3))
    S = np.vstack(out)
    keep = np.r_[True, np.linalg.norm(np.diff(S[:, :2], axis=0), axis=1) > 0.02]
    S = S[keep]
    if len(S) >= 5:
        from scipy.ndimage import median_filter
        S[:, 2] = median_filter(S[:, 2], size=5, mode='nearest')
    return _smooth_resample(S)


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
