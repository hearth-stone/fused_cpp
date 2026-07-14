#!/usr/bin/env python3

import argparse
import json
import pathlib
import subprocess
import sys


CASES = (
    ("tp4_w13_window", "w13", 4096, 512, (8, 16, 32, 64)),
    ("tp4_w13_full", "w13", 4096, 1024, (8, 16, 32, 64, 96)),
    ("tp4_w2", "w2_bf16", 512, 4096, (8, 16, 32, 64, 96)),
    ("ep4_w13_window", "w13", 4096, 2048, (8, 16, 32, 64, 96)),
    ("ep4_w13_full", "w13", 4096, 4096, (8, 16, 32, 64, 96)),
    ("ep4_w2", "w2_bf16", 2048, 4096, (8, 16, 32, 64, 96)),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep the standalone elastic SVE fused GEMM experiment"
    )
    parser.add_argument(
        "--binary",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("bench_elastic_nsplit"),
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--m", type=int, default=2040)
    parser.add_argument("--epoch-rows", type=int, default=204)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=9)
    parser.add_argument("--copies", type=int, default=4)
    parser.add_argument("--cpu-start", type=int, default=0)
    parser.add_argument(
        "--case",
        action="append",
        help="Run only a named case; may be specified more than once",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Use 3 iterations and 2 data copies"
    )
    return parser.parse_args()


def run_case(args, case_name, stage, k, n, threads):
    command = [
        str(args.binary),
        "--stage",
        stage,
        "--m",
        str(args.m),
        "--k",
        str(k),
        "--n",
        str(n),
        "--threads",
        str(threads),
        "--low-threads",
        str(max(1, threads // 2)),
        "--epoch-rows",
        str(args.epoch_rows),
        "--warmup",
        str(1 if args.quick else args.warmup),
        "--iters",
        str(3 if args.quick else args.iters),
        "--copies",
        str(2 if args.quick else args.copies),
        "--cpu-start",
        str(args.cpu_start),
    ]
    print(f"\n### {case_name} threads={threads}", flush=True)
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    print(completed.stdout, end="", flush=True)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)

    records = []
    for line in completed.stdout.splitlines():
        if line.startswith("RESULT_JSON ") or line.startswith("SUMMARY_JSON "):
            kind, payload = line.split(" ", 1)
            record = json.loads(payload)
            record["record_type"] = kind.removesuffix("_JSON").lower()
            record["case"] = case_name
            records.append(record)
    return records


def main():
    args = parse_args()
    selected = set(args.case or ())
    unknown = selected - {case[0] for case in CASES}
    if unknown:
        raise SystemExit(f"unknown cases: {', '.join(sorted(unknown))}")
    if not args.binary.exists():
        raise SystemExit(f"benchmark binary does not exist: {args.binary}")

    records = []
    for case_name, stage, k, n, thread_counts in CASES:
        if selected and case_name not in selected:
            continue
        for threads in thread_counts:
            records.extend(run_case(args, case_name, stage, k, n, threads))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"wrote {len(records)} records to {args.output}")

    summaries = [r for r in records if r["record_type"] == "summary"]
    print(
        "\ncase                 T  fragment%  phase%  strict%  "
        "elastic/static%  elastic/phase%"
    )
    print("--------------------------------------------------------------------------------")
    for record in summaries:
        print(
            f"{record['case']:<20} {record['threads']:>2} "
            f"{record['fragmentation_pct']:>10.2f} "
            f"{record['phase_claim_over_fixed_pct']:>7.2f} "
            f"{record['strict_claim_over_fixed_pct']:>8.2f} "
            f"{record['elastic_phase_over_piecewise_static_pct']:>16.2f} "
            f"{record['elastic_phase_over_piecewise_phase_pct']:>14.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
