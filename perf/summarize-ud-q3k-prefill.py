#!/usr/bin/env python3
"""Compare complete mirrored long-prefill arms, keeping independent samples."""
import argparse
import json
import statistics
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
parser.add_argument("prompts", type=int, nargs="+")
args = parser.parse_args()
print("| Tokens | Native seconds | Q3_K SoA seconds | Latency reduction | Native samples | SoA samples |")
print("|---:|---:|---:|---:|---|---|")
for prompt in args.prompts:
    groups = {}
    for arm in ("plain", "soa"):
        samples = []
        for rep in (1, 2):
            path = args.directory / f"pp{prompt}-{arm}-{rep}.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            assert len(rows) == 1, path
            row = rows[0]
            assert row["n_prompt"] == prompt and row["n_gen"] == 0, row
            assert row["n_ubatch"] == 512 and row["n_batch"] == 2048, row
            assert row["type_k"] == row["type_v"] == "turbo4", row
            samples.extend(ns/1e9 for ns in row["samples_ns"])
            route = path.with_suffix(".log").read_text()
            suffix = "1" if arm == "soa" else "0"
            pipeline = "kernel_mul_mm_n64_q3_K_f16_bci=0_bco=0_ne12=1_ne13=1_r2=1_r3=1_soa=" + suffix
            assert pipeline in route, (path, "missing route proof")
        groups[arm] = samples
    p, s = (statistics.mean(groups[arm]) for arm in ("plain", "soa"))
    pp, ss = (", ".join(f"{v:.3f}" for v in groups[arm]) for arm in ("plain", "soa"))
    print(f"| {prompt} | {p:.3f} | {s:.3f} | {100*(1-s/p):+.3f}% | {pp} | {ss} |")
print("\nPositive reduction means less time. Samples exclude full-length warmup and model loading.")
print("Two fresh processes per arm are not a precise confidence interval for sub-percent effects.")
