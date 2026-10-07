"""Combine sheet-labeled symbols with harvested ones into the final symbol bank.
Harvested lists are sorted best-first; per-class caps keep only the trustworthy head."""
import pickle
KEEP = {"(": 60, ")": 40, "[": 25, "]": 60, "{": 1, ",": 60, "-": 60, "<": 0, "1": 80, "2": 80, "7": 10, "8": 60, "=": 80, "≤": 40, "≥": 3, "+": 0, "/": 0, "5": 0, "3": 25, "6": 0, "9": 25, "4": 0, "0": 25, "×": 0}
if __name__ == "__main__":
    sheet = pickle.load(open("data/derived/symbols.pkl", "rb")); H = pickle.load(open("data/derived/symbols_harvested.pkl", "rb"))
    out = {}
    for c in set(sheet) | set(H):
        h = H.get(c, [])[:KEEP.get(c, 0)]
        sh = [u for u in sheet.get(c, []) if u.get("single")] or sheet.get(c, [])
        out[c] = (h + sh) if len(h) >= 5 else (sh + h)   # harvested first when there are enough of them
    # a zero is a closed loop: drop open U/V shapes that the harvest let through
    import numpy as np
    def closed(u):
        st = max(u["pieces"], key=len); gap = np.hypot(*(st[0, :2] - st[-1, :2])); h = st[:, 1].max() - st[:, 1].min()
        return gap < 0.45 * h
    if "0" in out: out["0"] = [u for u in out["0"] if closed(u)] or out["0"]
    for bad in ("4", "5", "6"):  # sheet instances for these are mislabeled or too few; better a gap than a wrong digit
        out.pop(bad, None)
    from inkclone.augment import augment_bank
    import random
    out = augment_bank(out, random.Random(11), per_instance=10, target_min=30, cap=150, strength=0.6)
    pickle.dump(out, open("data/derived/symbols_final.pkl", "wb"))
    print("final symbol bank:", {c: len(v) for c, v in sorted(out.items()) if v})
