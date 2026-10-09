# inkclone

Render typed documents in one person's handwriting, learned from their GoodNotes exports.

Input: GoodNotes PDF exports (Apple Pencil ink as vector ribbons, plus GoodNotes' invisible OCR
layer for word positions). Output: a vector-ink PDF (one stroked path per pen stroke, the writer's
pen width, sharp at any zoom) plus PNG previews.

Every character on an output page is one of the writer's own strokes, or a small deformation of
one (slant, size, bend, tail, dot position), or a composition of the writer's own parts (a j from an
i and a hook, an H from two stems and a bar). Words are assembled letter by letter; no word, chunk
or font is ever pasted as a unit.

## Usage

    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    cp "<GoodNotes export>.pdf" data/raw/              # any number of notebooks
    cp "<typed source>.txt" data/sources/<name>.txt    # optional, see "typed sources"
    .venv/bin/python -m inkclone.cli build             # rebuild every bank (~10 min for 90 pages)
    .venv/bin/python -m inkclone.cli render typed.pdf  # -> out/typed_handwritten.pdf + _pN.png

Typed sources: if `data/sources/<notebook>.txt` exists, that notebook is labeled by fuzzy
alignment of OCR word positions to the known text instead of trusting OCR text. Use it for a page
that was copied from a known document (a dictated sheet, homework from its typed solution).

## Pipeline (`inkclone/`)

| step | module | what it does |
|---|---|---|
| 1 | `extract` | GoodNotes ribbon pieces -> pen path + width per stroke (cap-split centerline, centroid-oriented chaining, arc-length smoothing, exact ribbon width) |
| 2 | `align`, `label_source` | OCR boxes -> labeled words (dictionary + line-context filters, or a typed source) |
| 3 | `segment`, `recut` | words -> letters: DP over cut points with width priors, then template-guided re-cut |
| 4 | `bank`, `drift` | style statistics: x-height, slant, pen width, gaps, join rates, page drift |
| 5 | `units` | letter bank: stroke-count sanity, curated blocklist, template-score tier, geometric plausibility, composed letters (j x X H D), deformation variants (`augment`) |
| 6 | `symbols`, `harvest_unlabeled`, `symbols_merge` | symbol and digit bank from a labeled symbol block plus shape-matched strokes from the unlabeled corpus, curated the same way |
| 7 | `source_doc` | typed PDF -> headers, pseudocode (with indentation), prose, display math, fractions, sub/superscripts |
| 8 | `page`, `synth2` | layout in the writer's conventions (two-column HW: code left, notes right), per-line slant/size/pen coherence, measured drift, letter-by-letter assembly, vector export |

## Curation

The glyph bank is reviewed by eye, under one rule: an instance stays if it is recognisably its
character, messy or not; it goes if it is incomplete, a fragment, or another character.

    .venv/bin/python scripts/curation_grid.py      # reports/curation_{lower,upper,digits,symbols}.png
                                                   # + data/curation/grid_labels.json (grid label -> source key)

Bad instances go into `data/curation/blocklist.json` by source key (notebook, page, word, position),
so a rebuild cannot bring them back. Composed letters and variants are generated only from what
survives curation.

## Evaluation

    .venv/bin/python scripts/evaluate.py <seed>   # reports/side_by_side_hw5.png (real page vs generated),
                                                  # reports/pick_the_fake.png (+ sealed KEY)

## State (2026-10-09)

Pages render cleanly as vector ink; pseudocode, prose and headers are legible and vary between
occurrences. Known gaps: digits 4 5 6 have no usable instance (7, 9 one or two), a few capitals (F, I,
J, Q, Y) rest on one or two real instances, and the hand-curation pass has to be repeated when the
extractor changes. `reports/LOG.md` is the running measurement log with every finding and fix.

Data (`data/`) and renders (`out/`) are not committed. `reports/` holds the current curation grids and
evaluation images only.
