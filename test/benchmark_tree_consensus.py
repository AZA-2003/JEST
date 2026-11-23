"""Run as PYTHONPATH=. python test/benchmark_tree_consensus.py --nodes 100000.

Index and solve timings exclude tree text generation and serialization. A unary
chain isolates dependence on the number of intervening original edges. Peak RSS
includes the Python process, JEST imports, the index, and input/output buffers.
"""
import argparse
import json
import resource
import time
from JEST.CTree import TreeConsensusIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=10000)
    parser.add_argument("--groups", type=int, default=1000)
    parser.add_argument("--method", choices=["frechet", "exponential_mode"], default="frechet")
    parser.add_argument("--kernel-alpha", type=float, action="append",
                        help="Required for exponential_mode; repeat to reuse the same index at different decay rates")
    args = parser.parse_args()
    if args.method == "exponential_mode" and not args.kernel_alpha:
        parser.error("exponential_mode requires --kernel-alpha")
    if args.method == "frechet" and args.kernel_alpha:
        parser.error("--kernel-alpha requires exponential_mode")
    if args.nodes < 128 or args.groups < 1:
        parser.error("nodes must be at least 128 and groups must be positive")
    n = args.nodes
    tree = "("*n + "x:1{1}" + "".join(f"):1{{{i}}}" for i in range(2,n+1)) + ");"
    start = time.perf_counter()
    index = TreeConsensusIndex(tree)
    index_time = time.perf_counter()-start
    for k in [2, 128]:
        for spread in [False, True]:
            sample = [1+i*(n-1)//(k-1) if spread else 1+i for i in range(k)]
            ids = sample*args.groups
            distal = [.25]*len(ids)
            weights = [1/k]*len(ids)
            offsets = list(range(0,len(ids)+1,k))
            counts = [1]*args.groups
            for threads in [1,4]:
                for kernel_alpha in args.kernel_alpha or [None]:
                    start = time.perf_counter()
                    if args.method == "frechet":
                        results = index.solve_batch(ids, distal, weights, offsets, counts, threads, True)
                    else:
                        results = index.exponential_mode_batch(ids, distal, weights, offsets, counts, kernel_alpha, threads, True)
                    duration = time.perf_counter()-start
                    start = time.perf_counter()
                    json.dumps([[r.edge_num,r.distal_length,(r.kernel_score if args.method == "exponential_mode" else r.mean_squared_distance)] for r in results])
                    serialize = time.perf_counter()-start
                    stages = {stage: sum(getattr(r, stage+"_seconds") for r in results)
                              for stage in ["virtual_tree", "moment", "lookup", "verification"]}
                    print(json.dumps(dict(method=args.method, kernel_alpha=kernel_alpha, nodes=n, groups=args.groups, k=k, spread=spread, threads=threads,
                                          index_seconds=index_time, batch_seconds=duration, stage_worker_seconds=stages,
                                          groups_per_second=args.groups/duration, serialize_seconds=serialize,
                                          peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))



if __name__ == "__main__":
    main()
