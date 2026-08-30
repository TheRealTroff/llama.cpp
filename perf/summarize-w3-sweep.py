#!/usr/bin/env python3
"""Summarize width-3 projection logs and their model-weighted round cost."""

import argparse
import re
from collections import defaultdict
from statistics import mean


PROJECTIONS = (
    "ffn_gate_up",
    "ffn_down",
    "attn_output",
    "attn_qkv",
    "attn_gate",
    "attn_q",
)
CALLS_PER_ROUND = dict(zip(PROJECTIONS, (128, 64, 64, 48, 48, 16)))
ARM_RE = re.compile(r"^ARM=(\S+)")
NAME_RE = re.compile(r"MUL_MAT\(name=([^,]+)")
TIME_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?) us/run")


def parse(path):
    samples = defaultdict(list)
    order = []
    arm = None
    run = None
    pending_name = None

    with open(path, encoding="utf-8", errors="replace") as source:
        for line in source:
            arm_match = ARM_RE.match(line)
            if arm_match:
                arm = arm_match.group(1)
                if arm not in samples:
                    order.append(arm)
                run = {}
                samples[arm].append(run)
                pending_name = None
                continue

            if arm is None:
                continue
            name_match = NAME_RE.search(line)
            if name_match:
                pending_name = name_match.group(1)
            time_match = TIME_RE.search(line)
            if pending_name and time_match:
                run[pending_name] = float(time_match.group(1))
                pending_name = None

    complete = {}
    for name in order:
        complete[name] = [
            run for run in samples[name]
            if all(projection in run for projection in PROJECTIONS)
        ]
    return order, complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    args = parser.parse_args()
    order, samples = parse(args.log)

    rows = []
    for arm in order:
        runs = samples[arm]
        if not runs:
            continue
        averages = {key: mean(run[key] for run in runs) for key in PROJECTIONS}
        round_ms = sum(averages[key] * CALLS_PER_ROUND[key] for key in PROJECTIONS) / 1000
        rows.append((arm, len(runs), averages, round_ms))

    if not rows:
        parser.error("no complete performance samples found")
    baseline = rows[0][3]
    print("arm\tn\t" + "\t".join(PROJECTIONS) + "\tround_ms\tdelta_pct")
    for arm, count, averages, round_ms in rows:
        values = "\t".join(f"{averages[key]:.2f}" for key in PROJECTIONS)
        delta = 100 * (round_ms / baseline - 1)
        print(f"{arm}\t{count}\t{values}\t{round_ms:.3f}\t{delta:+.2f}")


if __name__ == "__main__":
    main()
