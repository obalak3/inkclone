# Feasibility check on `Multivariable Calculus.pdf` (GoodNotes export, 24 pages)

- Ink is vector: page 2 has 0 images, 731 filled paths, ~86k bezier segments. 22 of 24 pages are handwritten.
- Paths are filled outlines (pressure pen), ribbons ~0.5pt wide (p10 0.27, p90 0.74). Width = pressure proxy.
- One PDF path = one pen stroke, but split into 3-10 closed sub-loops. Stroke order is preserved by draw order.
- GoodNotes embeds an invisible OCR text layer (render mode 3) with word boxes. Prose is mostly right, equations are garbage.
- Centerline recovery test on "Distance" (rasterize 600dpi + skeletonize): clean, 7 components for 8 letters, matches the real joins ("ance" is one stroke).

![centerline test](2026-10-06-centerline-test-Distance.png)

Open question: target documents are math or prose? Math needs its own layout + symbol work.

## Extraction + alignment results (same day)

- Extractor (`inkclone/extract.py`): per-loop opposite-point pairing -> centerline + width. Page 2 re-render vs PDF render at 600dpi: IoU 0.81, recall 0.955, precision 0.85. Whole notebook: 16,987 strokes, 2.07M samples, ~1 min.
- Alignment (`inkclone/align.py`): OCR word boxes -> stroke groups. Dictionary filter alone: 629 words, 37/48 labels correct on a random sheet (all errors = math OCR'd into short words). Adding the line-trust filter (line must be mostly dictionary tokens): 272 words, 128 distinct, 47/48 correct (`reports/contact_sheet_words_seed1.png`). Every lowercase letter covered.
- Yield is now the bottleneck: ~12 clean words per page. More notebooks scale it linearly; prose-heavy notebooks will yield far more than math ones.
