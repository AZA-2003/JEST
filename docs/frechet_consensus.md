# Continuous Fréchet consensus

Both consensus entry points accept `method="frechet"`. The default remains
`"legacy"`. Fréchet consensus minimizes the expected **squared branch-length
distance** between a consensus attachment point and the input attachments.
Each logical query contributes equally after its placement weights are
normalized. Pendant lengths do not enter the objective. The answer can lie
inside any reference edge, including an edge with no input placements.

```python
from JEST import placement_consensus, subgenome_consensus

# One result per original placement record; preserves n or nm and multiplicities.
placements = placement_consensus("input.jplace", False, method="frechet")

# With no mapping, all names belong to one group named "consensus".
placements, retained = subgenome_consensus(
    "input.jplace", True, dest_path="results", method="frechet"
)

# Optional richer API result; ordinary return shapes stay unchanged.
result = subgenome_consensus(
    "input.jplace", False, method="frechet", return_diagnostics=True
)
print(result["diagnostics"][0]["mean_squared_distance"])
# result also contains placements and remaining_placements.
```

```bash
JEST placement-consensus -i input.jplace -w -o results --method frechet
JEST subgenome-consensus -i input.jplace -w -o results --method frechet \
    --genome-map '([^_]+)_(.+)' --num-threads 4 --get-error
```

When either `write_file=True` or `get_error=True`, the method also writes
`<input>_<placement|subgenome>_consensus.tsv`. Its columns are `query` (or `group`
for grouped consensus), `consensus_label`, `edge_num`, and `distal_length`.
Per-query output has one row per distinct name, including every name in `nm`;
grouped output has one row per group, in result order.

The consensus label is the original child-node label of the selected edge.
Unnamed nodes use `edge_<edge_num>` without modifying the reference tree.
Quoted Newick labels are decoded, and TSV quoting preserves tabs or other
special characters. Labels need not be unique; the original integer edge ID
and full-precision distal coordinate identify the attachment. At a vertex,
the label follows the canonical selected edge, not necessarily that vertex's
own label. `consensus_label` is also included in the returned diagnostics and
jplace metadata.

`get_error=True` writes `<input>_<placement|subgenome>_frechet.tsv`. Columns
include group identity, original edge number, distal coordinate, logical query
count, positive placement row count, mean squared distance, root mean squared
distance, and mean distance. MSE is the optimized score; mean distance is a
separate compatibility diagnostic. Diagnostics are also included in output
jplace metadata. `return_diagnostics=True` returns a dictionary containing
`placements` and `diagnostics`, plus `remaining_placements` for grouped
consensus. Without it, the existing list / pair-of-lists return shapes remain.

## Query identity and grouping

Each distinct name in `n` or `nm` is one logical query. A multi-name record
provides the same distribution to each name, and names are assigned to groups
independently. `nm` abundance does not affect the equal-query objective.
Duplicate names, including duplicates inside one record, are rejected.
Per-record consensus preserves the entire original name list and multiplicities;
it solves identical distributions once.

A mapping file contains exactly two tab-separated columns, `query` and `group`,
with no header. Every input name needs a mapping. Extra mapping names are
allowed; conflicting assignments are rejected. A regex must match the whole
name and capture the genome in group 1 and subgenome in group 2. Groups are
returned in first-seen order. With no mapping, all queries form one group.

`multi_copy=True` and feature columns in mapping files are rejected because
they conflict with equal-query weighting. `edge_error=True` is also rejected:
the new diagnostics measure continuous branch-length distance.

`filter_queries=False` bypasses **all** uncertainty and random-distribution
code on this path. `filter_queries=True` is currently rejected because the
legacy option includes post-filtering against node labels, which loses the
consensus coordinate. Supply an explicitly prefiltered jplace instead; weights
and group query counts are recomputed from its retained names. No legacy
uncertainty statistic is silently replaced with MSE.

## Coordinates, validation, and output

The parser retains original unsigned 64-bit edge IDs and reads lengths directly
as doubles. `fields` must uniquely name `edge_num`, `like_weight_ratio`, and
`distal_length`; their order is arbitrary. Distal length is measured from the
child endpoint. Nonfinite or negative weights, empty distributions, unknown
edges, malformed rows, and meaningful out-of-range coordinates are errors.
Only out-of-range endpoint roundoff up to `32 * double epsilon * edge_length`
is clamped; small positive edges and interior coordinates are not rounded away.

Zero-length connected components are contracted. Vertices use the smallest
original incident edge ID, including zero-length edges within their component.
A positive annotated root stem is represented by an edge to a synthetic parent;
a zero root annotation represents the root point. Positive unannotated root
stems are rejected. Quoted labels, including doubled quote escapes, and
multifurcations are supported. General Newick bracket comments are not supported
and are rejected. Parsing and traversal are iterative, including on deep combs.

The original annotated tree and nonconflicting top-level metadata are preserved.
Output fields are explicitly:

```text
edge_num, likelihood, like_weight_ratio, distal_length, pendant_length
```

There is one output row with weight 1. Coordinates use full double precision.
Likelihood is a documented zero placeholder, not an estimated sequence
likelihood. Pendant length is the equal-query mixture mean when the input has
that column, otherwise a documented zero placeholder. Input-specific extra
placement columns are dropped. Metadata under `frechet` describes these
conventions and contains the diagnostics.

Accumulated depths and moments use `long double` (whose precision is platform
dependent). Selection uses first-moment directional derivatives and nonnegative
complementary sums, avoiding cancellation in minimized second-moment values.
The final objective is independently recomputed at the serialized coordinate.
Inputs whose positive lengths or interior coordinates collapse at the available
root-depth precision, or whose objective exceeds double range, fail explicitly.
This is floating-point computation, not arbitrary-precision arithmetic.

## Build and test

Rebuild the extension after pulling these sources; an older `_CTree.so` does not
contain the new solver. Python 3.12+ is needed by the existing Python source.
SWIG and a C++17 compiler with OpenMP are required.

```bash
# Linux; optional argument can override the Python include directory.
PYTHON=python3 SWIG=swig CXX=g++ bash JEST/compile_CTree.sh

# macOS / Apple clang, after installing libomp:
PYTHON=python3 SWIG=swig CXX=clang++ OPENMP_PREFIX="$(brew --prefix libomp)" \
    bash JEST/compile_CTree.sh

python -m pip install .
python -m pytest test/test_frechet.py -q
PYTHONPATH=. python test/benchmark_frechet.py --nodes 100000
```

The build generates `JEST/CTree.py` and `JEST/cbinder/CTree_wrap.cxx`; do not edit
those generated files. The package includes the compiled extension and generated
proxy, and wheels carry an interpreter/platform tag. As before, `setup.py` does
not itself compile the extension. Install the existing runtime dependencies
(`typer`, `treeswift`, `numpy`, `scipy`) and test dependency `pytest` as needed.

Tests cover the analytic examples, 2,000 seeded random-tree comparisons against
an exhaustive original-segment oracle, root/subdivision invariance, zero edges,
positive root stems, deep chains, arbitrary field order, large integer IDs,
sub-1e-4 coordinates, grouping, validation, serial/parallel agreement, API/CLI
routing, and legacy wrapper compatibility.

## Index lifetime and complexity

`JEST.CTree.FrechetTreeIndex(tree)` is immutable and can solve successive numeric
batches without rebuilding the index. `solve_batch` accepts integer edge IDs,
double distal coordinates, positive double weights, group offsets, query counts,
and a positive thread count. Callers of this low-level interface must normalize
each logical query themselves; the Python consensus wrapper handles this policy.
Offsets begin at 0 and end at the row count, with no empty groups. Empty batches
use offsets `[0]` and return no results. The index owns its state; make a new one
when the reference changes. Batches can be released independently of the index.

The shared Euler-tour RMQ and binary-ancestor index costs O(n log n) time and
memory. A group with k positive rows uses O(k) local memory and
O(k log(k+1) + log n) time, including one final original-edge lookup. There are no
per-group reference-tree traversals, reference-sized scratch resets, or tree
copies. OpenMP parallelizes independent groups, writes preallocated result
slots, and reports captured worker errors after the parallel region.

The optional last `solve_batch` argument `profile=True` enables timing fields
on each result for virtual-tree construction, moment solving, final edge lookup,
and direct objective verification. These are worker elapsed times; their sum
across parallel workers is not batch wall time. Profiling defaults to off.

## Measured performance

Measured on 2026-11-20 using macOS, x86_64 Python 3.12 under Rosetta,
Apple clang, `-O3`, and OpenMP. Each batch contains 1,000 groups. The reference
is a chain; the table uses observations spread over its entire length. Timings
include SWIG numeric-buffer conversion and optional per-group profiling, but
exclude Python jplace parsing/normalization. These are single-run measurements,
not universal throughput claims.

| Reference edges | Index (ms) | k | Serial batch (ms) | 4-thread batch (ms) | Serial groups/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 4.9 | 2 | 10.34 | 2.98 | 96,702 |
| 10,000 | 4.9 | 128 | 577.73 | 161.96 | 1,731 |
| 100,000 | 47.0 | 2 | 10.38 | 2.79 | 96,379 |
| 100,000 | 47.0 | 128 | 546.13 | 151.20 | 1,831 |
| 300,000 | 146.3 | 2 | 10.32 | 2.93 | 96,886 |
| 300,000 | 146.3 | 128 | 555.58 | 158.02 | 1,800 |

For the 300,000-edge reference with k=128, summed serial worker times in
milliseconds per 1,000 groups were:

| Virtual-tree construction | Moments and selection | Original-edge lookup | Objective verification |
| ---: | ---: | ---: | ---: |
| 94.86 | 335.46 | 0.61 | 96.14 |

Peak process RSS (including Python imports and retained batch buffers) was 136 MiB at 10,000 edges, 184 MiB at 100,000 edges, 320 MiB at 300,000 edges. Diagnostic JSON serialization took 1.9–3.7 ms per 1,000 results.

The solve times stay approximately flat as intervening path length grows at
fixed k; index construction and index memory grow with reference size. Raw
measurements, including clustered observations, both thread counts, phase
worker times, and RSS, are in [frechet_benchmark.jsonl](frechet_benchmark.jsonl).
The benchmark script can reproduce these measurements on other machines.
