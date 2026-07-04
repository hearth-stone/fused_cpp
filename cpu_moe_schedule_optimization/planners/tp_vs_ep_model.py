#!/usr/bin/env python3
"""Model TP=4 (low comm) vs EP=4 (high comm) MoE-layer latency.

Compute is grounded in MEASURED single-node cost models:
  - TP shard  F=512  -> contention_async_aws_8c_sha46228bb_20260703.json
  - EP full   F=2048 -> contention_async_aws_8c_ep4_F2048_20260704.json
Communication is a parameterized collective model (bandwidth + latency):
  - TP: ring all-reduce of [T, H]           (independent of top_k)
  - EP: all-to-all dispatch + combine of [T*top_k, H]  (scales with top_k)

Assumptions (edit via CLI): 1 MoE layer, P-way group, each rank = one 8-core
node, inter-node comm; per-rank compute FLOPs identical, efficiency differs by
shape. Tokens replicated for TP, data-parallel (T/P per rank) for EP.

Usage:
  python tp_vs_ep_model.py --tokens 2048 --topk 6 --experts 256 --bw-gbps 25 --latency-us 5
  python tp_vs_ep_model.py --sweep topk --bw-gbps 25       # find the crossover
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cost_model"))
from phase_model import ContentionCostModel          # noqa: E402
from interval_planner import IntervalPlanner          # noqa: E402

PROF_DIR = os.path.join(os.path.dirname(__file__), "..", "cost_model", "profiles")
TP_PROF = os.path.join(PROF_DIR, "contention_async_aws_8c_sha46228bb_20260703.json")
EP_PROF = os.path.join(PROF_DIR, "contention_async_aws_8c_ep4_F2048_20260704.json")


def compute_ms(model, experts, cores):
    """Best-schedule makespan (ns->ms) for a rank's expert workload."""
    return IntervalPlanner(model, cores).plan(experts)["makespan_ns"] / 1e6


def allreduce_ms(bytes_msg, P, bw_intra_Bps, bw_inter_Bps, lat_s):
    """Hierarchical all-reduce on a 2-pair NUMA topology (P=4 = 2 pairs of 2):
    reduce-scatter intra-pair, all-reduce across pairs, all-gather intra-pair.
    Only the reduced HALF crosses the slow inter-pair link (each direction)."""
    M = bytes_msg
    intra = 2 * (M / 2) / bw_intra_Bps          # RS + AG within a pair
    inter = 2 * (M / 2) / bw_inter_Bps          # cross-pair all-reduce of the partial
    return (intra + inter + 4 * lat_s) * 1e3


def all2all_ms(bytes_out_per_rank, P, bw_intra_Bps, bw_inter_Bps, lat_s):
    """All-to-all on the 2-pair topology. A rank's outgoing S splits: 1/P to the
    pair-partner (fast link) and 2/P to the far pair. BOTH near ranks' far-traffic
    shares the single slow inter-pair link -> aggregate 4S/P crosses it. Dispatch
    + combine doubles it. The slow link is the bottleneck."""
    S = bytes_out_per_rank
    intra = (S / P) / bw_intra_Bps              # to pair-partner
    inter = (4 * S / P) / bw_inter_Bps          # both near ranks -> far pair, shared link
    return 2 * (intra + inter + (P - 1) * lat_s) * 1e3   # x2 dispatch+combine


def model_layer(T, topk, E, H, F_full, P, bw_intra, bw_inter, lat_s, dbytes, cores):
    tp, ep = ContentionCostModel(TP_PROF), ContentionCostModel(EP_PROF)
    R = T * topk                                   # total routes in the layer
    # --- TP=P: all experts, F shard, all routes (tokens replicated) ---
    rpe_tp = max(1, R // E)
    exp_tp = [(e, rpe_tp) for e in range(E)]
    c_tp = compute_ms(tp, exp_tp, cores)
    comm_tp = allreduce_ms(T * H * dbytes, P, bw_intra, bw_inter, lat_s)
    # --- EP=P: E/P experts, full F, R/P routes (tokens data-parallel) ---
    E_rank = max(1, E // P)
    rpe_ep = max(1, (R // P) // E_rank)
    exp_ep = [(e, rpe_ep) for e in range(E_rank)]
    c_ep = compute_ms(ep, exp_ep, cores)
    comm_ep = all2all_ms((T // P) * topk * H * dbytes, P, bw_intra, bw_inter, lat_s)
    return (c_tp, comm_tp, c_tp + comm_tp), (c_ep, comm_ep, c_ep + comm_ep)


def fmt(tag, comp, comm, tot):
    return "%-6s compute=%7.2f  comm=%7.2f  total=%7.2f ms" % (tag, comp, comm, tot)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--f-full", type=int, default=2048)
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--cores", type=int, default=80, help="cores per rank (=NUMA node)")
    ap.add_argument("--bw-intra", type=float, default=60.0, help="intra-pair NUMA GB/s (0-1, 2-3)")
    ap.add_argument("--bw-inter", type=float, default=20.0, help="cross-pair NUMA GB/s ({0,1}-{2,3})")
    ap.add_argument("--latency-us", type=float, default=1.0)
    ap.add_argument("--dtype-bytes", type=int, default=2)
    ap.add_argument("--sweep", choices=["topk", "tokens", "bw"], default=None)
    a = ap.parse_args()
    bwi = a.bw_intra * 1e9; bwx = a.bw_inter * 1e9; lat = a.latency_us * 1e-6

    def one(T, topk, bwx_Bps):
        return model_layer(T, topk, a.experts, a.hidden, a.f_full, a.ranks,
                           bwi, bwx_Bps, lat, a.dtype_bytes, a.cores)

    if a.sweep is None:
        tp, ep = one(a.tokens, a.topk, bwx)
        print("T=%d topk=%d E=%d H=%d F_full=%d P=%d  BW intra=%.0f inter=%.0f GB/s (cores/rank=%d)" %
              (a.tokens, a.topk, a.experts, a.hidden, a.f_full, a.ranks, a.bw_intra, a.bw_inter, a.cores))
        print("  [compute = 8-core-profile placeholder; needs 80-core reprofile]")
        print(" ", fmt("TP", *tp)); print(" ", fmt("EP", *ep))
        win = "TP" if tp[2] < ep[2] else "EP"
        print("  winner=%s  (EP/TP total = %.2fx)" % (win, ep[2] / tp[2]))
        return
    print("sweep %s  (BW intra=%.0f inter=%.0f GB/s, T=%d topk=%d E=%d) [compute=8c placeholder]" %
          (a.sweep, a.bw_intra, a.bw_inter, a.tokens, a.topk, a.experts))
    print("  %-8s %22s %22s   winner" % (a.sweep, "TP total(comp+comm)", "EP total(comp+comm)"))
    if a.sweep == "topk":   vals = [1, 2, 4, 6, 8, 12]
    elif a.sweep == "tokens": vals = [64, 256, 512, 1024, 2048, 4096]
    else:                    vals = [10, 20, 40, 60, 100, 200]
    for v in vals:
        T = v if a.sweep == "tokens" else a.tokens
        tk = v if a.sweep == "topk" else a.topk
        bx = (v * 1e9) if a.sweep == "bw" else bwx
        tp, ep = one(T, tk, bx)
        print("  %-8s %8.2f (%.1f+%.1f) %8.2f (%.1f+%.1f)   %s" %
              (v, tp[2], tp[0], tp[1], ep[2], ep[0], ep[1], "TP" if tp[2] < ep[2] else "EP"))


if __name__ == "__main__":
    main()
