# inkclone

Learn one person's handwriting from GoodNotes exports and generate new text in it.

Input: GoodNotes PDF exports (Apple Pencil ink as vector outlines + GoodNotes' invisible OCR text layer).
Pipeline (planned):

1. `extract`  PDF -> per-stroke centerlines with width (pressure proxy), stroke order preserved
2. `align`    snap OCR word boxes onto stroke groups -> (strokes, text) pairs; drop garbled math
3. `report`   style statistics: join rate per letter pair, slant, x-height drift, spacing
4. `train`    stroke-sequence model fine-tuned on the writer's data
5. `layout`   document text -> page of generated strokes, rendered with the same pen model

Data lives in `data/` and is not committed (personal notes, large files).

## Setup

    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    cp "<GoodNotes export>.pdf" data/raw/

## Findings so far

See `reports/`.
