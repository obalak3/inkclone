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
    for bad in ("4", "5", "6"):  # sheet instances for these are mislabeled or too few; better a gap than a wrong digit
        out.pop(bad, None)
    pickle.dump(out, open("data/derived/symbols_final.pkl", "wb"))
    print("final symbol bank:", {c: len(v) for c, v in sorted(out.items()) if v})
