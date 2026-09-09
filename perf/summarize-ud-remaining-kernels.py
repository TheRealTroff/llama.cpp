#!/usr/bin/env python3
"""Summarize the four balanced arms; positive change means lower latency."""
import argparse
import math
import re
import statistics
from pathlib import Path


def read_arm(path):
    text = path.read_text()
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    pattern = re.compile(
        r"MUL_MAT\(type_a=(\w+),type_b=f32,m=(\d+),n=(\d+),k=(\d+),"
        r"[^\n]*?ud_remaining=1\):.*?(\d+) runs -\s+([0-9.]+) us/run", re.S)
    rows = {}
    for match in pattern.finditer(text):
        quant, m, n, k, runs, us = match.groups()
        key = (quant.removesuffix("_soa"), int(m), int(k), int(n))
        if key in rows:
            raise ValueError(f"duplicate {key} in {path}")
        rows[key] = (float(us), int(runs))
    if len(rows) != 132:
        raise ValueError(f"{path}: {len(rows)} timings, expected 132")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    arms = [read_arm(args.directory / f"{name}.log") for name in ("plain-1", "soa-1", "soa-2", "plain-2")]
    keys = set(arms[0])
    if any(set(arm) != keys for arm in arms):
        raise ValueError("arm shape sets differ")
    rows = []
    print("| Type | M | K | N | Plain us | SoA us | Latency reduction | Plain repeat spread | SoA repeat spread |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in sorted(keys):
        plain = [arms[i][key][0] for i in (0, 3)]
        soa = [arms[i][key][0] for i in (1, 2)]
        p, s = statistics.mean(plain), statistics.mean(soa)
        gain = 100*(1-s/p)
        spread_p, spread_s = 100*(max(plain)-min(plain))/p, 100*(max(soa)-min(soa))/s
        rows.append((key, p, s, gain, spread_p, spread_s))
        print(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {p:.2f} | {s:.2f} | {gain:+.1f}% | {spread_p:.1f}% | {spread_s:.1f}% |")
    print("\nGeometric-mean latency reduction across distinct shapes (not a model speedup):\n")
    print("| Type | N=1 | N=4 | N=1..8 | N=9,32 | N=512 |")
    print("|---|---:|---:|---:|---:|---:|")
    for quant in sorted({key[0] for key in keys}):
        values = []
        for widths in ({1}, {4}, set(range(1, 9)), {9, 32}, {512}):
            ratios = [s/p for key, p, s, *_ in rows if key[0] == quant and key[3] in widths]
            values.append(f"{100*(1-math.exp(statistics.mean(map(math.log, ratios)))):+.1f}%")
        print(f"| {quant} | " + " | ".join(values) + " |")
    print("\nWithin-arm repetition spread is a diagnostic, not a confidence interval.")
    print(f"\nMaximum spread: plain {max(r[4] for r in rows):.1f}%, SoA {max(r[5] for r in rows):.1f}%.")
    print(f"Cases over 5% spread: plain {sum(r[4] > 5 for r in rows)}, SoA {sum(r[5] > 5 for r in rows)}.")
    print("\nRun counts and exact compiled pipeline names are retained in the raw logs.")


if __name__ == "__main__":
    main()
