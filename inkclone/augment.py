"""Natural variants of real glyph instances by small smooth deformations.

Every variant is built from one real instance: a low-frequency bend along the stroke,
a slightly longer or shorter tail, a nudged dot/cross, a little shear, scale and pen
weight. All deformations are gentle (a few percent of the x-height) so the letter
stays the writer's; the point is that the same letter never repeats identically.
"""
import numpy as np


def _arc_t(p):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1))]
    return d / d[-1] if d[-1] > 1e-9 else np.linspace(0, 1, len(p))


def variant(inst, rng, strength=1.0):
    pieces = [p.copy() for p in inst["pieces"]]
    allp = np.vstack([p[:, :2] for p in pieces]); h = max(allp[:, 1].max() - allp[:, 1].min(), 0.5)
    main_i = int(np.argmax([len(p) for p in pieces]))
    s = strength
    # global affine: anisotropic scale, shear, tiny rotation
    sx, sy = rng.uniform(1 - 0.06 * s, 1 + 0.06 * s), rng.uniform(1 - 0.06 * s, 1 + 0.06 * s)
    shear = np.tan(np.radians(rng.uniform(-3, 3) * s)); rot = np.radians(rng.uniform(-2, 2) * s)
    c, sn = np.cos(rot), np.sin(rot)
    for p in pieces:
        x, y = p[:, 0] * sx, p[:, 1] * sy
        x = x + y * shear
        p[:, 0], p[:, 1] = c * x - sn * y, sn * x + c * y
    # smooth bend along each stroke (two low harmonics), amplitude ~5% of the glyph height.
    # Near-straight strokes (bars, parentheses, 1, l) are left alone: bending them changes the glyph.
    for p in pieces:
        if len(p) < 6: continue
        chord = p[-1, :2] - p[0, :2]; L_ = np.linalg.norm(chord)
        if L_ > 1e-6:
            u_ = chord / L_; rel = p[:, :2] - p[0, :2]
            dev = np.abs(u_[0] * rel[:, 1] - u_[1] * rel[:, 0]).max()  # perpendicular distance from the chord
            if dev < 0.18 * L_: continue
        t = _arc_t(p); amp = 0.05 * h * s
        for k in (1, 2):
            ph_x, ph_y = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
            p[:, 0] += amp / k * np.sin(2 * np.pi * k * t * 0.5 + ph_x)
            p[:, 1] += amp / k * np.sin(2 * np.pi * k * t * 0.5 + ph_y)
    # tail: stretch or shrink the last 15% of the main stroke along its own direction
    m = pieces[main_i]
    if len(m) >= 10:
        n_tail = max(3, int(0.15 * len(m))); f = rng.uniform(0.8, 1.2)
        base = m[-n_tail - 1, :2]; m[-n_tail:, :2] = base + (m[-n_tail:, :2] - base) * f
    # diacritics (small pieces): nudge position, mostly upward/sideways
    for i, p in enumerate(pieces):
        ext = p[:, :2].max(0) - p[:, :2].min(0)
        if i != main_i and max(ext) < 0.45 * h:
            p[:, 0] += rng.uniform(-0.12, 0.12) * h * s; p[:, 1] += rng.uniform(-0.15, 0.08) * h * s
    # pen weight
    wf = rng.uniform(0.9, 1.1)
    for p in pieces: p[:, 2] *= wf
    # re-anchor: left edge at 0, baseline unchanged
    x0 = min(p[:, 0].min() for p in pieces)
    for p in pieces: p[:, 0] -= x0
    allq = np.vstack([p[:, :2] for p in pieces])
    out = dict(inst); out.update({"pieces": pieces, "width": float(allq[:, 0].max()), "height": float(allq[:, 1].max() - allq[:, 1].min()),
                                  "synthetic": True, "seed_src": inst.get("src")})
    return out


def augment_bank(bank, rng, per_instance=10, target_min=40, cap=200, strength=1.0):
    """Add variants so every glyph has at least `target_min` instances (up to `cap`).
    Glyphs with many real instances get a couple of variants each; rare ones get `per_instance`."""
    out = {}
    for c, insts in bank.items():
        real = [u for u in insts if not u.get("synthetic")]
        if not real: out[c] = list(insts); continue
        n_var = per_instance if len(real) < 8 else max(2, min(per_instance, (target_min - len(real)) // len(real) + 1)) if len(real) < target_min else 1
        new = list(real)
        for u in real:
            for _ in range(n_var):
                if len(new) >= cap: break
                new.append(variant(u, rng, strength))
        out[c] = new
    return out


def compose_j(i_inst, g_inst, rng):
    """A lowercase j from the writer's own parts: the i's stem and dot, with the stem continued
    below the baseline into the descender hook taken from a g (or y)."""
    pieces = [p.copy() for p in i_inst["pieces"]]
    main_i = int(np.argmax([len(p) for p in pieces])); stem = pieces[main_i]
    # stem bottom = the lowest point; orient the stem top-to-bottom
    if stem[0, 1] > stem[-1, 1]: stem = stem[::-1]
    bottom = stem[-1, :2].copy()
    g = max(g_inst["pieces"], key=len)
    below = g[g[:, 1] > 0.15]                                   # the g's descender part
    if len(below) < 6:                                          # parametric hook fallback
        ang = np.linspace(np.pi / 2, np.pi * 1.45, 14); r = 0.42
        hook = np.column_stack([bottom[0] - r + r * np.cos(ang) * -1, bottom[1] + 0.35 + r * np.sin(ang) - r, np.full(14, stem[:, 2].mean())])
    else:
        # take the descender from its topmost point downward, keep its shape, scale to ~0.9 xh deep
        start = int(np.argmin(below[:, 1])); seg = below[start:].copy() if start < len(below) - 6 else below.copy()
        depth = max(seg[:, 1].max() - seg[:, 1].min(), 1e-6); f = rng.uniform(0.8, 1.0) / depth
        seg[:, :2] = (seg[:, :2] - seg[0, :2]) * f + bottom
        seg[:, 2] = stem[:, 2].mean(); hook = seg
    new_stem = np.vstack([stem, hook[1:]])
    pieces[main_i] = new_stem
    x0 = min(p[:, 0].min() for p in pieces)
    for p in pieces: p[:, 0] -= x0
    allq = np.vstack([p[:, :2] for p in pieces])
    return {"text": "j", "pieces": pieces, "width": float(allq[:, 0].max()), "height": float(allq[:, 1].max() - allq[:, 1].min()),
            "src": ("composed", i_inst.get("src"), g_inst.get("src")), "synthetic": True, "whole": True, "score": 0.1,
            "slant": i_inst.get("slant"), "pen": i_inst.get("pen"), "joined_prev": False, "joined_next": False}


def compose_x(halves, rng):
    """An X from two single-stroke halves (a '\\' and a '/') the cutter separated: overlay them centred."""
    def slope(u):
        p = max(u["pieces"], key=len); d = p[-1, :2] - p[0, :2]
        return d[0] * d[1]  # >0: top-left to bottom-right ('\\'), <0: '/'
    back = [u for u in halves if slope(u) > 0]; fwd = [u for u in halves if slope(u) < 0]
    if not back or not fwd: return None
    a, b = rng.choice(back), rng.choice(fwd)
    pa = max(a["pieces"], key=len).copy(); pb = max(b["pieces"], key=len).copy()
    for p in (pa, pb):
        c = (p[:, :2].max(0) + p[:, :2].min(0)) / 2; p[:, :2] -= c
    h = max(pa[:, 1].max() - pa[:, 1].min(), pb[:, 1].max() - pb[:, 1].min(), 1e-6)
    pb[:, :2] *= (pa[:, 1].max() - pa[:, 1].min()) / max(pb[:, 1].max() - pb[:, 1].min(), 1e-6) * rng.uniform(0.95, 1.05)
    pieces = [pa, pb]
    x0 = min(p[:, 0].min() for p in pieces); ybot = max(p[:, 1].max() for p in pieces)
    for p in pieces: p[:, 0] -= x0; p[:, 1] -= ybot - 0.05   # sit on the baseline
    allq = np.vstack([p[:, :2] for p in pieces])
    return {"text": "X", "pieces": pieces, "width": float(allq[:, 0].max()), "height": float(allq[:, 1].max() - allq[:, 1].min()),
            "src": ("composed", a.get("src"), b.get("src")), "synthetic": True, "whole": True, "score": 0.1,
            "slant": a.get("slant"), "pen": a.get("pen"), "joined_prev": False, "joined_next": False}
