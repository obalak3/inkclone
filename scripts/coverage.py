"""How much of a typed source document can be synthesised from whole words, from
joined chunks (letter runs written as one pen stroke), and from single letters?"""
import sys, re, pickle, collections, numpy as np, pymupdf
sys.path.insert(0, "."); from inkclone.segment import load_words
src = sys.argv[1]
text = " ".join(p.get_text() for p in pymupdf.open(src))
tokens = re.findall(r"[A-Za-z]+", text)
d = pickle.load(open("data/derived/segments.pkl", "rb")); bank = pickle.load(open("data/derived/bank.pkl", "rb"))["bank"]
words = {w["text"].lower() for w in d["words"]}
words_cs = {w["text"] for w in d["words"]}
chunks = collections.Counter()
for w, seg in zip(d["words"], d["segs"]):
    if not seg: continue
    run = seg[0]["char"]
    for prev, ch in zip(seg, seg[1:]):
        if ch["joined_prev"]: run += ch["char"]
        else:
            if len(run) >= 2: chunks[run] += 1
            run = ch["char"]
    if len(run) >= 2: chunks[run] += 1
letters = {c for c, v in bank.items() if v}
def tile(t):
    """greedy longest-chunk tiling; returns (n_chunks_used, n_single_letters, missing letters)"""
    i, nc, nl, miss = 0, 0, 0, []
    while i < len(t):
        for L in range(min(6, len(t) - i), 1, -1):
            if chunks.get(t[i:i+L], 0) >= 1: nc += 1; i += L; break
        else:
            nl += 1; miss += [t[i]] if t[i] not in letters else []; i += 1
    return nc, nl, miss
cat = collections.Counter(); missing = collections.Counter(); examples = collections.defaultdict(list)
for t in tokens:
    if t.lower() in words: cat["whole word"] += 1; continue
    nc, nl, miss = tile(t)
    if miss: cat["has letters NOT in bank"] += 1; missing.update(miss); examples["missing"].append(t)
    elif nl == 0: cat["chunks only"] += 1; examples["chunks"].append(t)
    else: cat["chunks + single letters"] += 1; examples["mixed"].append(t)
n = len(tokens); print(f"{src}: {n} word tokens, {len(set(x.lower() for x in tokens))} distinct")
for k, v in cat.most_common(): print(f"  {k:28s} {v:4d}  {v/n:5.1%}")
print("letters missing from bank:", missing.most_common())
print("whole-word bank:", len(words), "distinct;", "chunk bank:", len(chunks), "distinct chunks, top:", chunks.most_common(15))
print("examples needing letters:", sorted(set(examples["mixed"]))[:25])
print("examples with missing letters:", sorted(set(examples["missing"]))[:25])
w = np.concatenate([p[:, 2] for insts in bank.values() for i in insts for p in i["pieces"]]); print(f"bank pen width now: median {np.median(w):.2f}pt")
