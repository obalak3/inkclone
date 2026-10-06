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

## Segmentation, bank, first synthesis (same day, later)

- Corpus after adding 8 notebooks: 82 handwritten pages, 69,527 strokes, 1,249 labeled words (`scripts/run_all.py`).
- Character segmentation (`inkclone/segment.py`): DP over cut points with learned per-letter width priors. 5,112 letter instances. Measured join behaviour: 31% of adjacent pairs joined overall, "an" 85%, "er" 74%, "nd" 2%, "ed" 3"; join INTO d/b/w 3%/8%/10%.
- Style: x-height 3.9pt median (IQR 3.0..4.9), slant -4.4 deg, unjoined letter gap 0.19 xh, word gap 2.2 xh (2,007 pairs).
- Glyph bank with per-class height filter: 3,967 instances. Per-letter grid (`reports/bank_grid_aetDs.png`): only ~65-80% of instances for common letters are clean cuts, the rest carry neighbour fragments. THIS is the current quality bottleneck.
- First synthesis (`inkclone/synth.py`, `reports/compare_line_v2.png`): recognisably the same letterforms, not passable yet. Causes: dirty instances (above), pen width overestimated 0.66 vs 0.49pt (fixed in extractor: area/perimeter width), sparse capitals (D: 1 clean instance).
- Next: shape-based outlier rejection per letter, then template-guided re-segmentation (EM), then re-evaluate the grid before touching the synthesiser again.

## Capitals sheet + source-text labeling (evening)

- `inkclone/label_source.py`: labels a page from its typed source via fuzzy sequence alignment of OCR boxes to source tokens (1-2 boxes per token). Capitals sheet: 255/281 tokens matched, 244 words, 48/48 labels correct on the contact sheet.
- Segmentation DP vectorised: 211s -> 13s for the corpus.
- Corpus: 1,493 labeled words, 6,169 letter instances. Every capital has >= 4 raw instances.
- Capital filter bug: writer's capitals are 3.5-4.5 x-heights; range widened to 1.8-5.5. Capital width prior raised to 1.8.
- Whole-word outlier filter (width / stroke count vs same-word median): 1,489 -> 1,321 word instances.
- `reports/compare_line_v5.png`: whole-word lines good; assembled words improved (B, F present) but still rough. Letter cuts around capitals are the dirtiest. Next: template-guided re-cutting.
- Template-guided re-cut (`inkclone/recut.py`, vectorised, 28s for the corpus): 953/1493 words changed cuts. e/t/s grids visibly cleaner (`units_letter_grid_aets_v2.png`), "a" less so.
- Letters that are complete pen strokes (unjoined both sides) are clean by construction: 2,247 of 3,598. Synth2 now prefers them.
- `compare_line_v7.png`: whole-word lines good except occasional mislabeled word instances (an OCR "the" that is really "that"). Assembled words still inconsistent word to word.
- Template scoring of every unit (letters: 2nd-nearest template distance; words and chunks: mean letter score). Drop worst 15% of word instances (1355 -> 1166) and 219 chunks. Quality-weighted sampling in Synth2 (softmax over score, never argmin, so variation stays).
- `compare_line_v9.png`: "Shortest Paths with Few Steps", "Bellman Ford, stop when a round", "No change round means fixed point, so stopping early is safe" all readable and in the writer's hand. Still weak: "changes", "nothing" (chunk 'ng' from a poor source).

## Symbols (loop tick, ~18:40)

- `inkclone/symbols.py`: pairs handwritten lines to source lines by fuzzy match on OCR letters+digits (the writer skipped one source line, so positional pairing was off by one), then aligns ink clusters to space-separated tokens, and only cuts inside multi-char tokens.
- Result (`reports/symbols_grid.png`): `[ ] ( ) = + < ≤ ≥ ∞ ∈` and digits mostly clean; 7/8/9 perfect. Commas and the single-instance set symbols (← ∪ ∩ ∅) unreliable. Enough for a first homework page; a second short symbol line or two would fix the rest.

## Layout engine v1 (loop tick, ~19:05)

- `inkclone/source_doc.py` parses the typed PDF: front matter, numbered headers, pseudocode blocks with indentation, prose paragraphs (merged across PDF line breaks), sub/superscripts marked _ ^.
- `inkclone/page.py` lays out GoodNotes-size pages in the writer's HW conventions and renders PNG + PDF. First full `set05.pdf` rendering: `reports/page_set05_v3_p1.png`, `_p2.png`, `.pdf`.
- Reads as the writer's hand at a glance; structure correct. Weak layer = symbols/punctuation: `( ) | . , -` instances are dirty or few, `⇒ → ⇔` missing (writer draws ⇒ as "=>"), digits unreliable. Next: harvest single-stroke symbols/punctuation from the 69k unlabeled strokes by shape-matching against the few clean seeds, and compose ⇒/→ from parts.
