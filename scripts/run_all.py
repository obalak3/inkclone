"""Extract + align every GoodNotes PDF in data/raw, print a per-notebook yield table."""
import sys, glob, os, time, json
sys.path.insert(0, ".")
from inkclone.extract import extract_pdf, save
from inkclone.align import load, align, save_samples
rows = []
for src in sorted(glob.glob("data/raw/*.pdf")):
    name = os.path.splitext(os.path.basename(src))[0]; prefix = "data/derived/" + name
    t = time.time()
    if not os.path.exists(prefix + ".npz"):
        save(extract_pdf(src), prefix)
    pages = load(prefix)
    if os.path.exists(f"data/sources/{name}.txt"):
        from inkclone.label_source import label_pages
        samples, _, _ = label_pages(pages, open(f"data/sources/{name}.txt").read())
    else:
        samples, _ = align(pages)
    save_samples(samples, prefix + ".words")
    n_pages = sum(1 for p in pages if p["n_strokes"] > 0); n_strokes = sum(p["n_strokes"] for p in pages)
    rows.append((name, n_pages, n_strokes, len(samples), len({s["text"].lower() for s in samples}), time.time() - t))
    print(f"{name:28s} pages={n_pages:3d} strokes={n_strokes:6d} words={len(samples):4d} distinct={rows[-1][4]:4d} ({rows[-1][5]:.0f}s)", flush=True)
tot = sum(r[3] for r in rows); print(f"TOTAL labeled words: {tot}")
