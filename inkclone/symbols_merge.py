"""Combine sheet-labeled symbols with harvested ones into the final symbol bank.
Harvested lists are sorted best-first; per-class caps keep only the trustworthy head."""
import pickle
KEEP = {"(": 60, ")": 40, ".": 40, "[": 25, "]": 60, "{": 1, ",": 60, "-": 60, "<": 0, "1": 80, "2": 80, "7": 10, "8": 60, "=": 80, "≤": 40, "≥": 3, "+": 0, "/": 0, "5": 0, "3": 0, "6": 0, "9": 0, "4": 0, "0": 25, "7": 0, "×": 0}
if __name__ == "__main__":
    sheet = pickle.load(open("data/derived/symbols.pkl", "rb")); H = pickle.load(open("data/derived/symbols_harvested.pkl", "rb"))
    out = {}
    for c in set(sheet) | set(H):
        h = H.get(c, [])[:KEEP.get(c, 0)]
        sh = sheet.get(c, []) if c.isdigit() else ([u for u in sheet.get(c, []) if u.get("single")] or sheet.get(c, []))
        out[c] = (h + sh) if len(h) >= 5 else (sh + h)   # harvested first when there are enough of them
    # a zero is a closed loop: drop open U/V shapes that the harvest let through
    import numpy as np
    def closed(u):
        st = max(u["pieces"], key=len); gap = np.hypot(*(st[0, :2] - st[-1, :2])); h = st[:, 1].max() - st[:, 1].min()
        return gap < 0.45 * h
    if "0" in out: out["0"] = [u for u in out["0"] if closed(u)] or out["0"]
    import json, os
    if os.path.exists("data/curation/blocklist.json"):
        block = {tuple(k) for k in json.load(open("data/curation/blocklist.json"))}
        for c in list(out):
            out[c] = [u for u in out[c] if tuple(u["src"]) + (None,) not in block]
            if not out[c]: del out[c]
    # commas and periods are small; a tall "comma" is a mis-harvested 3 or 9
    for c in ",.":
        if c in out: out[c] = [u for u in out[c] if u["height"] <= 1.25] or out[c]
    # parentheses and brackets are tall and narrow; a wide '(' reads as a C
    for c in "()[]":
        if c in out:
            out[c] = [u for u in out[c] if u["width"] / max(u["height"], 1e-6) <= 0.42] or out[c]
    # digits: only the hand-picked sheet seeds (data/curation/digit_seeds.json) plus harvested instances
    seeds = {c: {tuple(k) for k in v} for c, v in json.load(open("data/curation/digit_seeds.json")).items()}
    for c in "0123456789":
        if c in out:
            out[c] = [u for u in out[c] if u.get("harvested") or tuple(u["src"]) in seeds.get(c, set())]
            for u in out[c]:  # digits are one stroke (4 and 5 up to two); anything extra is a stray fragment
                keep_n = 2 if c in "45" else 1
                if len(u["pieces"]) > keep_n:
                    u["pieces"] = sorted(u["pieces"], key=len, reverse=True)[:keep_n]
                    allq = np.vstack([p[:, :2] for p in u["pieces"]]); x0 = allq[:, 0].min()
                    for p in u["pieces"]: p[:, 0] -= x0
                    u["width"] = float(allq[:, 0].max() - x0); u["height"] = float(np.ptp(allq[:, 1]))
            out[c] = [u for u in out[c] if u["height"] >= 1.1 or tuple(u["src"]) in seeds.get(c, set())]  # hand-picked seeds are trusted
            if not out[c]: del out[c]
    from inkclone.augment import augment_bank
    import random
    out.pop(")", None)  # always mirrored from the writer's clean '(' at render time
    # Σ: the sheet's only instance is a flat cut; build it as a stroke with the writer's pen instead
    from inkclone.augment import compose_sigma
    if not any(u["height"] > 1.0 for u in out.get("Σ", [])):
        pen = float(np.median([u.get("pen") or np.median(np.concatenate([p[:, 2] for p in u["pieces"]])) for u in out.get("=", [])[:20]]) or 0.65)
        out["Σ"] = [compose_sigma(pen, random.Random(21 + k)) for k in range(12)]
    out = augment_bank(out, random.Random(11), per_instance=10, target_min=30, cap=150, strength=0.6, bend=False)
    pickle.dump(out, open("data/derived/symbols_final.pkl", "wb"))
    print("final symbol bank:", {c: len(v) for c, v in sorted(out.items()) if v})
