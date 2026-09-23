"""Summaries for the executed-shell results.

    python scripts/analyze_llm.py shutdown results/llm/ds32b_block.json results/llm/ds32b_nonblock.json
        -> per-frame executed/intent tampering with Wilson intervals; Fisher tests for the blocking
           contrast (within framing) and the framing contrast (within blocking condition)
    python scripts/analyze_llm.py restart results/llm/ds32b_L0.json ... results/llm/ds32b_L4.json
        -> per-level counts (by framing and pooled) and the one-sided Cochran-Armitage trend test
"""
import sys, json
from _common import ROOT
from llm.analyze import cell, fisher, cochran_armitage, load_rows


def shutdown(block_path, nonblock_path):
    b, n = load_rows(block_path), load_rows(nonblock_path)
    frames = [f for f in ("control", "recovery", "recovery_strong") if any(r["frame"] == f for r in b)]
    print(f"{'frame':16s} {'exec block':>18s} {'exec non-block':>18s} {'intent block':>18s}")
    for f in frames:
        sb = [r for r in b if r["frame"] == f]; sn = [r for r in n if r["frame"] == f]
        cb, cn, ib = cell(sb), cell(sn), cell(sb, "tamper_intent")
        fmt = lambda c: f"{c['k']}/{c['n']} {c['ci']}"
        print(f"{f:16s} {fmt(cb):>18s} {fmt(cn):>18s} {fmt(ib):>18s}")
    print("\nFisher exact (executed tampering):")
    for f in frames:
        print(f"  blocking vs non-blocking, {f}: p = {fisher([r for r in b if r['frame']==f], [r for r in n if r['frame']==f])}")
    if "recovery_strong" in frames:
        print(f"  recovery_strong vs control, blocking: p = {fisher([r for r in b if r['frame']=='recovery_strong'], [r for r in b if r['frame']=='control'])}")


def restart(paths):
    files = [json.load(open(p)) for p in paths]
    files.sort(key=lambda d: int(d["level"]) if str(d["level"]).isdigit() else {"keep": 0, "wipe": 3}[d["level"]])
    frames = sorted({r["frame"] for d in files for r in d["rows"]})
    print(f"{'level':6s} " + " ".join(f"{f:>16s}" for f in frames) + f" {'pooled':>12s}")
    counts, ns = [], []
    for d in files:
        rows = d["rows"]; lv = d["level"]
        line = f"L{lv!s:5s} "
        for f in frames:
            c = cell([r for r in rows if r["frame"] == f]); line += f" {c['k']}/{c['n']} {c['ci']}".rjust(17)
        c = cell(rows); line += f" {c['k']}/{c['n']}".rjust(13); counts.append(c["k"]); ns.append(c["n"]); print(line)
    if len(files) >= 3:
        z, p = cochran_armitage(counts, ns)
        print(f"\nCochran-Armitage trend over levels (pooled framings): z = {z:.2f}, one-sided p = {p:.2g}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    if sys.argv[1] == "shutdown":
        shutdown(sys.argv[2], sys.argv[3])
    else:
        restart(sys.argv[2:])
