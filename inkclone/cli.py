"""inkclone command line.

  python -m inkclone.cli build                 rebuild every bank from data/raw (+ data/sources labels)
  python -m inkclone.cli render typed.pdf      -> out/<name>_pN.png, out/<name>.pdf, out/<name>.vector.pdf
"""
import sys, os, glob, subprocess, time


def run(mod, *args):
    t = time.time(); print(f"-> {mod} {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable, "-m", mod, *args], capture_output=True, text=True)
    out = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "deprecated" not in l and l.strip())
    print("   " + out.replace("\n", "\n   ")[-1500:], f"({time.time()-t:.0f}s)")
    if r.returncode: sys.exit(f"{mod} failed")


def build():
    os.makedirs("data/derived", exist_ok=True)
    run("scripts.run_all") if os.path.exists("scripts/__init__.py") else subprocess.run([sys.executable, "scripts/run_all.py"], check=True)
    gaps = lambda: subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0,'.'); from inkclone.synth import measure_word_gaps; measure_word_gaps()"], check=True)
    # pass 1: width-prior cuts -> first letter bank (gives recut its templates)
    run("inkclone.segment"); run("inkclone.bank"); gaps(); run("inkclone.units")
    # pass 2: template-guided re-cut -> final banks
    run("inkclone.recut"); run("inkclone.bank"); gaps(); run("inkclone.drift"); run("inkclone.units")
    run("inkclone.symbols"); run("inkclone.harvest_unlabeled"); run("inkclone.symbols_merge")


def render(src, out=None, seed=1):
    os.makedirs("out", exist_ok=True)
    out = out or os.path.join("out", os.path.splitext(os.path.basename(src))[0] + "_handwritten")
    run("inkclone.page", src, out, str(seed))
    print("written:", out + ".pdf", out + ".vector.pdf", "and", out + "_p*.png")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("build", "render"): sys.exit(__doc__)
    if sys.argv[1] == "build": build()
    else: render(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None, int(sys.argv[4]) if len(sys.argv) > 4 else 1)
