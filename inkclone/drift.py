"""Page-level handwriting drift statistics from the writer's own pages (OCR word boxes):
line pitch variation, within-line baseline slope, and left-margin wander."""
import glob, json, pickle
import numpy as np
from .label_source import reading_order


def lines_of(words):
    lines = []
    for w in sorted(words, key=lambda w: (w["bbox"][1], w["bbox"][0])):
        y0, y1 = w["bbox"][1], w["bbox"][3]
        for ln in lines:
            ly0, ly1 = ln["y"]
            if min(y1, ly1) - max(y0, ly0) > 0.4 * min(y1 - y0, ly1 - ly0):
                ln["ws"].append(w); ln["y"] = (min(y0, ly0), max(y1, ly1)); break
        else:
            lines.append({"ws": [w], "y": (y0, y1)})
    return sorted(lines, key=lambda ln: ln["y"][0])


def measure(derived_glob="data/derived/*.json"):
    pitch_ratio, slopes, margins = [], [], []
    for f in glob.glob(derived_glob):
        if f.endswith(".words.json"): continue
        for pg in json.load(open(f)):
            lines = [ln for ln in lines_of(pg["words"]) if len(ln["ws"]) >= 3]
            if len(lines) < 4: continue
            ys = np.array([np.median([w["bbox"][3] for w in ln["ws"]]) for ln in lines])  # bottoms ~ baseline
            pitches = np.diff(ys); med = np.median(pitches)
            if med <= 0: continue
            pitch_ratio += [p / med for p in pitches if 0.5 < p / med < 2.0]
            for ln in lines:
                xs = np.array([(w["bbox"][0] + w["bbox"][2]) / 2 for w in ln["ws"]]); yb = np.array([w["bbox"][3] for w in ln["ws"]])
                if xs.max() - xs.min() > 80:
                    slopes.append(float(np.polyfit(xs, yb, 1)[0]))  # pt of baseline drop per pt of x
            x0s = np.array([min(w["bbox"][0] for w in ln["ws"]) for ln in lines]); left = x0s[x0s < np.percentile(x0s, 40) + 15]
            if len(left) >= 3: margins += list(left - np.median(left))
    st = {"pitch_ratio_sd": float(np.std(pitch_ratio)), "pitch_ratio_iqr": [float(np.percentile(pitch_ratio, 25)), float(np.percentile(pitch_ratio, 75))],
          "baseline_slope_sd": float(np.std(slopes)), "baseline_slope_median": float(np.median(slopes)),
          "margin_sd_pt": float(np.std(margins)), "n": (len(pitch_ratio), len(slopes), len(margins))}
    return st


if __name__ == "__main__":
    st = measure(); print(st)
    d = pickle.load(open("data/derived/bank.pkl", "rb")); d["stats"]["drift"] = st; pickle.dump(d, open("data/derived/bank.pkl", "wb"))
