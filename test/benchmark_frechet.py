"""Run as PYTHONPATH=. python test/benchmark_frechet.py --nodes 100000.

Index and solve timings exclude tree text generation and serialization. A unary
chain isolates dependence on the number of intervening original edges. Peak RSS
includes the Python process, JEST imports, the index, and input/output buffers.
"""
import argparse
import json
import resource
import time
from JEST.CTree import FrechetTreeIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=10000)
    parser.add_argument("--groups", type=int, default=1000)
    args = parser.parse_args()
    n = args.nodes
    tree = "("*n + "x:1{1}" + "".join(f"):1{{{i}}}" for i in range(2,n+1)) + ");"
    start = time.perf_counter()
    index = FrechetTreeIndex(tree)
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
                start = time.perf_counter()
                results = index.solve_batch(ids, distal, weights, offsets, counts, threads, True)
                duration = time.perf_counter()-start
                start = time.perf_counter()
                json.dumps([[r.edge_num,r.distal_length,r.mean_squared_distance] for r in results])
                serialize = time.perf_counter()-start
                stages = {stage: sum(getattr(r, stage+"_seconds") for r in results)
                          for stage in ["virtual_tree", "moment", "lookup", "verification"]}
                print(json.dumps(dict(nodes=n, groups=args.groups, k=k, spread=spread, threads=threads,
                                      index_seconds=index_time, batch_seconds=duration, stage_worker_seconds=stages,
                                      groups_per_second=args.groups/duration, serialize_seconds=serialize,
                                      peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))


if __name__ == "__main__":
    main()
