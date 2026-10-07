# inkclone

Learn one person's handwriting from GoodNotes exports and render typed documents in it.

Input: GoodNotes PDF exports (Apple Pencil ink arrives as vector outlines, plus GoodNotes'
invisible OCR text layer). Output: a vector-ink PDF (one filled ribbon per stroke, pen width preserved, opens as ink,
sharp at any zoom) plus PNG previews rasterised from it.

## Usage

    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    cp "<GoodNotes export>.pdf" data/raw/            # any number of notebooks
    cp "<typed source>.txt|.pdf" data/sources/       # optional: exact text of a page you copied (see below)
    .venv/bin/python -m inkclone.cli build           # ~2.5 min for 90 pages: all banks rebuilt
    .venv/bin/python -m inkclone.cli render typed.pdf   # -> out/typed_handwritten.pdf (vector ink) + _pN.png previews

Typed sources: if `data/sources/<notebook name>.txt` exists, that notebook is labeled by fuzzy
alignment of OCR word positions to the source text instead of trusting OCR text. Use it for pages
that were copied from a known document (homework from its typed solution, a dictated sheet).

## How it works (pipeline, all in `inkclone/`)

1. `extract`   PDF ribbons -> per-stroke centerline + width, pen order preserved
2. `align` / `label_source`   OCR word boxes -> labeled words (dictionary + line-context filters, or typed source)
3. `segment` + `recut`   words -> letters: DP over cut points with width priors, then template-guided
4. `bank`, `units`   style statistics; whole-word / joined-chunk / letter banks with quality scores and geometric plausibility checks
5. `symbols`, `harvest_unlabeled`, `symbols_merge`   symbol and digit bank from a labeled symbol block plus shape-matched strokes from the unlabeled corpus
6. `source_doc`   typed PDF -> headers / pseudocode / prose / sub-superscripts
7. `page`   layout in the writer's conventions (two-column HW: code left, notes right), per-line slant / size / pen coherence, measured drift; `synth2` picks units longest-first (whole word > chunk > letter)

## State (2026-10-06)

- Whole words and pseudocode read as the writer's hand. Assembled words are close; headers and long
  rare words are the roughest register. See `reports/side_by_side_hw5.png`, `reports/pick_the_fake.png`.
- Symbols: `( ) [ ] = + < ≤ ≥ ∞ ∈ , -` and digits 0 1 2 7 8 solid; `⇒ → ⇔` composed from parts;
  3 and 9 thin; 4 5 6 absent (one written line of spaced digits fixes this).
- Everything is data-bound: more prose and heading-rich exports raise whole-word coverage directly.

Data in `data/` and renders in `out/` are not committed. `reports/` holds the measurement log
(`2026-10-06-feasibility.md`) and the evaluation images.
