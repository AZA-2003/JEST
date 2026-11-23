# Exponential proximity consensus

Use `method="exponential_mode"` to maximize

\[
S(c)=\frac{\sum_j a_j\exp(-\alpha d(c,x_j))}{\sum_j a_j}.
\]

As with Fréchet consensus, each query's placement weights are normalized
independently and every logical query contributes equally. The full
placement distributions are pooled. Distances are between attachment points;
pendant lengths do not contribute.

This score is expected exponential proximity, **not** a normalized density on
the reference tree or an estimated sequence likelihood. `kernel_alpha` is
required, finite, and positive. Its units are inverse branch length, and
`1 / kernel_alpha` is the distance at which proximity drops to `exp(-1)`.
There is no automatic bandwidth choice. The existing `alpha` argument still
belongs to filtering and is not the kernel parameter. Supplying `kernel_alpha`
for another method raises an error.

```python
from JEST import placement_consensus, subgenome_consensus

placements = placement_consensus(
    "input.jplace", False, method="exponential_mode", kernel_alpha=2.0,
)
result = subgenome_consensus(
    "input.jplace", True, dest_path="results", method="exponential_mode",
    kernel_alpha=2.0, return_diagnostics=True,
)
print(result["diagnostics"][0]["kernel_score"])
```

```bash
conda activate jest
JEST subgenome-consensus -i input.jplace -w -e -o results \
    --method exponential_mode --kernel-alpha 2.0
```

Both entry points retain their existing return shapes. Rich diagnostics add
`kernel_alpha`, `kernel_score`, `kernel_tie_count`, and `kernel_tie_tolerance`.
The optimized score is `kernel_score`; mean distance and MSE remain comparison
diagnostics. If an optional distance diagnostic exceeds floating-point range,
it is `null` in JSON/Python output (blank in TSV); the bounded kernel score
remains usable. At the low-level C++ interface, unavailable distance diagnostics
are positive infinity.

The output jplace has the same five standard fields as Fréchet output, with
placement weight 1 and a zero likelihood placeholder. It preserves the original
tree and reports the same equal-query pendant-length convention. Method metadata
is stored under `metadata.exponential_mode`, including the decay rate, objective,
and tie policy. Stale metadata under the other indexed consensus method is
removed when reprocessing its output; unrelated metadata is preserved.

The consensus jplace and label TSV retain their usual filenames:
`<input>_<placement|subgenome>_consensus.jplace` and
`<input>_<placement|subgenome>_consensus.tsv`. With `get_error=True`, diagnostics
are written to `<input>_<placement|subgenome>_exponential_mode.tsv`.

Name handling, `n`/`nm` multiplicities, file/regex grouping, row validation,
original integer edge IDs, zero-length components, and positive root stems all
follow the [shared consensus conventions](frechet_consensus.md). `multi_copy`
and feature-weighted mappings remain unsupported. Fréchet pre/post filtering,
including the `filter_queries` alias, is **not supported** for this method.
Explicit filtering thresholds and old null-distribution files are also rejected.
The Fréchet null distributions are not calibrated for exponential-mode centers.
Filtering remains available with `method="frechet"`.

## Modes, ties, and precision

Unlike the Fréchet mean, this method can have multiple global maxima. Two
separated equally weighted observations have modes at their endpoints, not
their midpoint. Unequal weights favor the heavier endpoint. For three equal
masses on unit star arms, the center wins when `exp(-kernel_alpha) > 1/2`;
the tips win when it is below `1/2`, and all four candidates tie at equality.

The implementation first finds the numerical maximum over all virtual-tree
candidates, then considers candidates within
`64 * epsilon(long double) * total_weight` of that maximum tied. It selects the
smallest `(canonical original edge ID, distal coordinate)` among those candidates.
It never averages tied positions. A group supported at a single geometric point
returns that point and score 1 directly. The reported normalized tolerance is
`kernel_tie_tolerance`; `kernel_tie_count` counts the qualifying **virtual
candidates**, including retained original endpoints. A numerical tie is not a
proof of an exact mathematical tie, nor an enumeration of every point within
tolerance in the continuous tree.

Messages and verification use compensated sums and `long double` accumulators.
Root distances retain low-order corrections so that small local lengths are
not lost merely by subtracting large root depths. Original observed distal
coordinates are preserved for final serialization. The existing checks against
unrepresentable lengths/interior coordinates still apply. `long double` precision
is platform-dependent; on Apple Silicon it has double precision. For extremely
small decay rates, score differences can fall below working precision and are
reported as numerical ties. Very large decay-distance products may underflow
attenuation to zero; the local observation mass still contributes at its point.
The selected score is independently recomputed at the serialized coordinate and
checked against its message score.

## Implementation and complexity

The parser, immutable index, numeric batch validation, OpenMP worker/error
handling, virtual-tree builder, query normalization, grouping, and output path
are shared with Fréchet consensus. The Fréchet quadratic/descent optimizer is
unchanged. Exponential mode uses its own two message passes and scans every
virtual vertex. On a segment without an observation or branching point the
score is convex, so its maximum is at an endpoint; there is no continuous
edge optimization, sampling grid, or nonlinear search.

The retained provenance makes candidate tie comparisons and final serialization
O(1), with no ancestor search per candidate or at the winner. The shared index
costs O(n log n) time/memory. Per group, virtual-tree construction costs
O(k log(k+1)); the message passes, candidate scan, and direct score verification
each cost O(k), with O(k) working memory. No full-reference traversal occurs
inside a group solve.

The low-level reusable API is:

```python
from JEST.CTree import TreeConsensusIndex
index = TreeConsensusIndex(annotated_newick)
results = index.exponential_mode_batch(
    edge_ids, distal_lengths, positive_weights, group_offsets, query_counts,
    kernel_alpha, threads, profile=False,
)
```

It shares the `ConsensusResult` container and SWIG result-vector
conversion; kernel-specific fields are populated only by exponential mode.
The positional signature and behavior of `solve_batch` remain intact. Repeated
calls with different decay rates reuse the same index; attenuation is local to
each solve and is never cached across rates. Low-level callers must perform their
own within-query normalization, as for `solve_batch`.

Rebuild with [the existing build instructions](frechet_consensus.md#build-and-test).
Generated SWIG files remain generated. Run the full regression suites with:

```bash
python -m pytest test/test_frechet.py test/test_frechet_filtering.py \
    test/test_exponential_mode.py -q
PYTHONPATH=. python test/benchmark_tree_consensus.py --method exponential_mode \
    --nodes 100000 --kernel-alpha 0.001 --kernel-alpha 1 --kernel-alpha 100
```

The tests include analytic endpoint/star cases, same-edge interior observations,
true and numerical ties, tiny/large decay rates, zero components, root stems,
weight/row invariance, rooting/subdivision invariance, 2,000 independently scored
random original trees, high-degree serial/parallel agreement, validation,
API/CLI output, and existing Fréchet/filtering regressions.

## Measured performance

Measured on 2026-11-23 with native Apple Silicon Python 3.12, Apple clang,
`-O3`, and OpenMP. Each batch has 1,000 groups and profiling enabled. The table
uses placements spread along a chain, with `kernel_alpha=1`. Timings include
SWIG numeric-buffer conversion and exclude jplace parsing/normalization.
These single-run observations are not universal throughput guarantees.

| Reference edges | Index (ms) | Rows/group | Serial batch (ms) | 4-thread batch (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 1.9 | 2 | 1.39 | 0.43 |
| 10,000 | 1.9 | 128 | 54.42 | 19.34 |
| 100,000 | 19.5 | 2 | 1.36 | 0.42 |
| 100,000 | 19.5 | 128 | 55.37 | 18.54 |
| 300,000 | 57.6 | 2 | 1.34 | 0.44 |
| 300,000 | 57.6 | 128 | 59.08 | 18.34 |

Each size reused one index across decay rates 0.001, 1, and 100, clustered and
spread observations, and both thread counts. The fixed-k solve time remains
approximately flat as the number of intervening original edges grows. Peak
process RSS, including Python imports and retained input/output buffers, was
144 MiB at 10,000 edges, 192 MiB at 100,000 edges, 340 MiB at 300,000 edges.

[Raw measurements](exponential_mode_benchmark.jsonl) include per-stage worker
times for construction, message passes/selection, direct serialization, and
objective verification, plus batch throughput and JSON serialization time.
Worker times summed across parallel workers are not batch wall times.
