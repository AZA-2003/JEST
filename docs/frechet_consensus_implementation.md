# Implementing weighted tree consensus in JEST

Implement an exact consensus attachment point by minimizing expected squared path distance on the reference tree. Preserve the full placement distribution of each query, normalize it independently, and give every query in a consensus group equal total weight. Use a shared tree index and a query-local virtual tree so the computation does not traverse the full reference tree for every consensus.

The original Frechet specification below refers to JEST commit `139e3541b3eee717c752cb5a58096d5ffac1f58c`, inspected on 2026-11-20. Frechet code has subsequently been added; consult `docs/frechet_consensus.md` for its implemented behavior rather than treating the original integration observations as the current state. The final section, added on 2026-11-23, specifies an exponential-kernel mode extension; that section describes proposed work, not an implemented feature. Source paths below are relative to the repository root so the handoff remains portable.

## Agreed objective and statistical interpretation

For a consensus group containing q logical queries, query i has attachment points x_ij and nonnegative placement weights w_ij. Normalize each query so that sum_j w_ij = 1. The objective is

\[
 F(c)=\frac1q\sum_{i=1}^{q}\sum_j w_{ij}d(c,x_{ij})^2,
 \qquad \widehat c=\arg\min_{c\in T}F(c).
\]

Distances are ordinary branch-length distances between attachment points on the reference tree. Pendant lengths are not part of these distances. The center may be inside any edge or at any vertex; it is not restricted to observed attachments.

The interpretation is: choose a query uniformly, then choose its true attachment according to that query's placement distribution. The consensus minimizes expected squared attachment error. This is the tree Frechet mean. It does not require a diffusion model, a bandwidth, or an exponential decay rate. Do not describe it as the maximum-likelihood center of a normalized exponential or Gaussian density on a finite tree.

For implementation, pool the original placements with effective weights a_ij = w_ij/q. The common factor 1/q may be omitted when finding the location, but normalize reported mean squared distance by total effective weight. Never first replace each query by its own mean: means of query means need not equal the pooled mean on a branching tree.

Example: query A has one placement at position 0 on a line; query B has placements at 8 and 12, each of weight 1/2. Effective weights are 1/2, 1/4, 1/4, and the consensus is 5. Counting placement rows equally would incorrectly give 20/3.

## Complexity and a correction to the earlier shorthand

Let n be the reference-tree size, k the total number of positive-weight placement rows in one group before merging duplicates, and K the number of vertices in its constructed virtual tree. K = O(k).

| Operation | Time | Additional memory |
| --- | --- | --- |
| Parse topology, edge IDs, and coordinates | O(n) | O(n) |
| Euler-tour sparse-table LCA index | O(n log n) | O(n log n) |
| Binary ancestors for output position lookup | O(n log n) | O(n log n) |
| Normalize and group placement rows | O(k) after group membership is known | O(k) |
| Construct one virtual tree | O(k log(k+1)) | O(k) |
| Moment passes and continuous minimization | O(k) | O(k) |
| Locate winning point on an original edge | O(log n), once per consensus | O(1) working memory |
| Directly verify final objective | O(k) with point-distance queries | O(1) working memory |

The straightforward end-to-end bound is O(k log(k+1) + log n) per consensus, not strictly O(k log k) when original edge-number output is included. The optimization itself has the desired O(k log k) bound. The extra weighted-ancestor lookup is small and avoids walking a long compressed path. Do not add an O(log n) lookup for every candidate.

For many groups, the tree index is built once. With p active workers, memory is O(n log n) plus their query-local buffers and input/output storage. There is no n-by-n distance matrix, O(n) buffer clearing per query, or dependence on the number of original edges between placements during optimization.

If binary lifting is also used for LCA, instead of constant-time Euler-tour RMQ, state the additional O(k log n) query term honestly. Root depths alone do not make LCA constant time.

## Existing JEST integration points

| Source and symbol | Existing behavior relevant to this change | Required treatment |
| --- | --- | --- |
| `JEST/JEST.py`, `placement_consensus` | Reads jplace, creates internal labels, calls `CTree.placement_consensus` | Add an explicit method selector and route the new method through precise point coordinates |
| `JEST/JEST.py`, `subgenome_consensus` | Groups queries, performs uncertainty filtering, calls C++ consensus | Apply equal-query pooling after the selected filtering policy; preserve group identities |
| `JEST/cbinder/CTree.cpp`, `get_placement_consensus` | Finds edge labels by repeated string searches, selects a label using the old score | Bypass for the new method |
| `JEST/cbinder/CTree.cpp`, `get_subgenome_consensus` | Aggregates by node label; optional mapping features and multi-copy adjustments | Reuse validated group identities only; do not reuse its aggregation or weighting |
| `JEST/cbinder/CTree.cpp`, `find_length_pc`, `find_length_sc` | Fits attachment lengths separately, including randomized initialization | Never call for Frechet consensus; the optimizer already returns the point |
| `JEST/cbinder/CTree.cpp`, consensus output wrappers | Serialize a selected label, then round attachment lengths to four decimals | Return exact edge identity and full double-precision distal length |
| `JEST/cbinder/CTree.cpp`, `placement_stat_subgenome` | Uses consensus labels and node distances | Cannot be used unchanged for point-aware Frechet diagnostics or post-filtering |
| `JEST/cbinder/CTree_utils.h`, `placement_obj` | Stores placement rows in `vector<vector<float>>` | Use a separate double-precision input path or migrate this representation and all bindings |
| `JEST/cbinder/CTree.i`, input typemap | Explicitly casts Python numeric inputs to float | Avoid this conversion for the new method; widening afterwards does not recover precision |
| `JEST/cbinder/compact_tree.h`, `calc_dist` | Walks ancestors and creates hash maps | Use the shared index, not this method in query loops |
| `JEST/cbinder/compact_tree.h`, length macros | Defaults to float; supports `CT_LENGTH_DOUBLE` | Parse lengths as doubles from the outset; apply build macros consistently if reusing this type |
| `JEST/compile_CTree.sh` | Compiles explicit source list and regenerates SWIG wrapper | Add new compilation units and declarations; keep generated files generated |

Two wrapper details need explicit handling. First, `subgenome_consensus` currently computes a null distribution and performs an initial filtering loop before its later `if filter_queries` block. Do not claim that setting `filter_queries=False` currently bypasses all filtering. For the new method, make selection of contributing queries explicit and skip filtering when disabled. Second, post-filtering currently passes labels to `placement_stat_subgenome`. Either implement a point-aware version with the existing statistic's intended semantics, or reject that optional combination clearly. Do not silently substitute MSE into an uncertainty statistic calibrated for a different quantity.

## Proposed public behavior

Add `method="legacy" | "frechet"` to both Python consensus entry points and expose `--method` in the CLI. Keep the existing default initially for compatibility. The same Frechet solver handles a single query and a group of queries; only pooling differs.

Suggested new components are `JEST/cbinder/frechet_consensus.h` and `.cpp`, containing a reusable immutable `FrechetTreeIndex`, a small `WeightedPoint` representation, and a batch solver. These are proposed files, not existing interfaces.

Prefer a typed input boundary to the new solver:

```cpp
struct WeightedPoint {
    uint32_t edge_index;  // internal dense index, not a floating-point edge ID
    double distal;
    double weight;
};

struct FrechetResult {
    uint64_t edge_num;    // original external jplace edge ID
    double distal_length;
    double mean_squared_distance;
    double mean_distance; // optional compatibility diagnostic
    size_t query_count;
    size_t positive_row_count;
};
```

Python should resolve jplace field columns and logical-query membership once, then pass numeric buffers and group offsets through new SWIG declarations/typemaps. Validate integer IDs before converting to the internal dense index. Keep `placement_obj` on the legacy path if a global representation migration would broaden the change unnecessarily. Return edge IDs to Python as integers and all real values as doubles.

Build one index per input reference tree per batch. If a persistent index is later exposed, make its ownership explicit and never reuse it after topology, lengths, or edge IDs change. Do not cache by the caller's mutable object identity alone.

## Query identity and equal contribution

Normalize every contributing logical query independently before pooling. Reject negative or nonfinite weights, and reject a query with no positive total weight by default. Drop zero-weight rows. If filtering intentionally removes a query, recompute q from the retained queries. An empty group should produce a structured error or explicitly omitted result, never a fabricated root placement.

The current C++ code frequently uses `placement.n[0]`. Do not silently carry that assumption into the new path. Proposed policy: each distinct sequence name is one logical query; a record with multiple names supplies the same placement distribution to each name. Resolve each name's group separately. For `nm` records, use each name once and ignore abundance multiplicity for the equal-query objective; document this policy. Reject duplicate names across records unless an explicit merge rule is implemented. Preserve names and multiplicities when returning per-query results.

These are serialization policies to document and test, not additional biological weighting. If the application instead treats a multi-name record as one logical query, expose that as a separate declared policy rather than inferring it from row count.

For mapping files and regex grouping, parse group membership before the numerical solver. Validate missing names, unmatched regexes, required capture groups, malformed rows, and conflicting assignments. With no mapping, all supplied queries form one group, consistent with the documented grouped-consensus behavior.

The agreed objective does not include feature-column weights or a third layer of multi-copy weighting. For the initial `frechet` method, reject `multi_copy=True` and explicit feature-weighted mapping inputs with a message explaining the conflict, rather than inheriting the existing adjustments or silently ignoring requested weighting. A future hierarchical mode could split a logical query's unit weight among its copies before splitting among placements, but that is a separate contract.

## Parsing and point coordinates

Use original edge numbers as stable identities. Parse the annotated Newick tree once and build `edge_num -> (parent, child, length)`. Do not identify an edge by substring matching a node label, assume edge numbers are dense, or assume topology traversal indices equal external edge numbers. Preserve the original output tree and its edge annotations.

For a parent u, child v, edge length L, and jplace distal length delta, the coordinate from u is

\[
 t=L-\delta,\qquad D(x)=D(u)+t=D(v)-\delta.
\]

Distal length is measured from the child/distal endpoint, not the rootward endpoint. A jplace record can contain several names, or names with multiplicities, and field order comes from `fields`. These conventions are defined in [A Format for Phylogenetic Placements](https://pmc.ncbi.nlm.nih.gov/articles/PMC3284489/).

Require `edge_num`, `like_weight_ratio`, and `distal_length`; never guess their column positions. Validate row sizes, finite lengths, unique edge IDs, nonnegative tree lengths, and 0 <= delta <= L. Permit only a small scale-aware endpoint tolerance for parsing roundoff; reject meaningful out-of-range values instead of projecting them silently. Canonicalize endpoints to vertex identities and merge coincident placements by summing effective weights. Different distal positions on the same edge must remain distinct.

Use a tokenizer/parser that retains edge annotations, labels, and numeric lengths together. If the existing parser is reused, attach edge IDs through a synchronized structural parse, with an explicit one-to-one topology check; repeated string scans are not an acceptable substitute. Parse iterative traversals or otherwise avoid recursion failure on comb-shaped trees.

Handle zero-length edges as a metric quotient: contract connected zero-length components for optimization, retain a deterministic map back to original edge endpoints, and merge all placements at the same contracted point. Do not contract small positive edges just because they are below a display tolerance. Audit any reused helper's zero thresholds; `compact_tree::add_child` has a `ZERO_THRESH` check. If the whole tree contracts to one point, return its canonical representation with objective zero.

Root annotations need an explicit convention. A root edge with positive stem length must be represented as an actual edge to a synthetic proximal vertex if it is part of the placement domain. If unsupported initially, reject placements on it clearly. Do not silently discard a positive root stem or reinterpret its distal coordinate. A zero-length root annotation can represent the root point.

## Shared tree index

Store parent, children, edge lengths, original edge identities, DFS entry/exit times, and root distances. Use an iterative DFS to produce the Euler tour, topological depths, and first-occurrence indices.

Build an RMQ sparse table over Euler-tour topological depths for O(1) LCA. Use topological depth for LCA, not weighted root distance. Root distances supply path lengths once the LCA is known. Build binary ancestors as well for locating the final point on an ancestor path by cumulative length.

All this state is immutable during batch solving. Use doubles at minimum; compensated sums and/or long doubles for accumulated depths and moments are useful when long root paths contain tiny local edges. No precision can be recovered after float parsing. Nonnegative path lengths obtained by subtraction may require a roundoff check; a substantially negative value is an error.

For arbitrary edge-interior point distances, avoid walking ancestors. If both points are on the same original edge, their distance is the absolute difference of their distal coordinates. Otherwise, take the minimum over the four endpoint pairs of: distance from the first point to its endpoint, indexed vertex distance between endpoints, and distance from the second endpoint to the second point. This is O(1) with O(1) LCA and is a useful independent helper for final verification.

## Constructing the virtual tree with edge-interior observations

Use this simple construction rather than extending LCA to artificial point nodes:

1. For every positive-weight placement on an edge, collect both original endpoints. Canonicalized vertex observations also mark their vertex. Work on the zero-length-contracted topology.
2. Sort the collected vertices by DFS entry time and deduplicate.
3. Add the LCAs of consecutive vertices in that order. Sort and deduplicate again.
4. Build parent links using an ancestor stack: pop until its top is an ancestor of the next vertex, then link the next vertex to that top. Its virtual-edge length is the difference of weighted root distances.
5. For each occupied original edge, insert its interior placement points, sorted from parent to child. Both endpoints were collected, so this original edge is present explicitly before subdivision. Replace it with the ordered chain of points and endpoints.
6. Merge exact/canonical coincident points, assign their effective masses, and generate preorder/postorder arrays of the resulting O(k)-vertex tree.

The retained original endpoints can add a few segments outside the minimal connecting subtree. This is harmless: the virtual tree still has O(k) vertices, and the true minimizer belongs to the smaller subtree. There is no need to expand any compressed path into original edges.

Every virtual vertex needs a location descriptor: original vertex ID or original edge ID plus offset. Every virtual edge connects ancestor-related locations and preserves the full path length. Save enough provenance to locate a winning interior point later. Never mutate the shared reference tree to insert query points.

## Why optimizing on the virtual tree is exact

Let H be the smallest subtree connecting the positive-weight observations. For a point outside H, moving to its projection onto H decreases every distance, hence decreases the squared-distance objective. Some minimizer lies in H.

Between consecutive virtual vertices, no placement or path leading to an additional placement branches off. Removing the segment interior partitions the observations into two sets. Moving distance s along the segment increases distances to one set by s and decreases distances to the other by s. Therefore the objective is a quadratic with positive leading coefficient equal to total weight.

Unobserved side branches are irrelevant to this expected-loss objective. Unlike the normalized density models considered earlier, there is no Z(c), ball-volume term, or topology-dependent penalty to account for them.

Squared distance is strictly convex along any nonzero tree geodesic after summing with positive total weight. Equivalently, distance to each observation along a geodesic has the form a + |s-b|; its square has positive quadratic curvature and a nonnegative derivative jump. Thus the geometric minimizer is unique on the contracted tree. Multiple edge labels may describe that same point at a vertex.

## Moment equations and exact edge minimization

Root the virtual tree arbitrarily, for example at its topmost retained vertex. Let a[v] be mass at v, and W be total mass (normally one). For the virtual subtree rooted at v define

\[
 m_v=\sum_{x_j\in T_v}a_j,\qquad
 h_v=\sum_{x_j\in T_v}a_jd(v,x_j),\qquad
 q_v=\sum_{x_j\in T_v}a_jd(v,x_j)^2.
\]

For each child z at distance ell, postorder accumulation is

\[
 m_v=a[v]+\sum_z m_z,
\]
\[
 h_v=\sum_z(h_z+\ell m_z),
 \qquad
 q_v=\sum_z(q_z+2\ell h_z+\ell^2m_z).
\]

Mass at v contributes zero to its first and second distance moments.

Let H[v] and Q[v] be first and second moments over all observations. At root r, H[r]=h[r] and Q[r]=q[r]. For an edge u -> v of length ell, put

\[
 B=h_v+\ell m_v.
\]

B is the first distance moment measured from u of observations on the v-side. Rerooting gives

\[
 H[v]=H[u]+\ell(W-2m_v),
\]
\[
 Q[v]=Q[u]+2\ell(H[u]-2B)+W\ell^2.
\]

At distance s from u along this virtual edge,

\[
 Q_e(s)=Q[u]+2s(H[u]-2B)+Ws^2.
\]

Its exact constrained minimizer is

\[
 \boxed{s_e=\operatorname{clip}\left(\frac{2B-H[u]}{W},0,\ell\right).}
\]

Scan all virtual edges, compute this candidate and its objective, and choose the minimum, also initializing the best candidate to the root. For a single virtual vertex, return it directly. This costs O(K). No iterative gradient search, random initialization, or edge sampling is needed.

### Reference pseudocode

```text
solve(group, index):
    points = normalize_each_query_then_pool(group)
    V = build_virtual_tree(points, index)
    root, preorder, postorder = orient(V)

    for v in postorder:
        mass[v] = observation_mass[v]
        down1[v] = down2[v] = 0
        for (child, length) in children[v]:
            mass[v] += mass[child]
            down1[v] += down1[child] + length * mass[child]
            down2[v] += down2[child] + 2*length*down1[child]
                        + length*length*mass[child]

    W = mass[root]
    require W > 0
    all1[root] = down1[root]
    all2[root] = down2[root]
    best = (root, all2[root])

    for u in preorder:
        for (v, length) in children[u]:
            B = down1[v] + length * mass[v]
            t = (2*B - all1[u]) / W
            s = clamp(t, 0, length)
            value = all2[u] + 2*s*(all1[u] - 2*B) + W*s*s
            update_best(best, virtual_edge=(u,v), offset=s, value=value)
            all1[v] = all1[u] + length*(W - 2*mass[v])
            all2[v] = all2[u] + 2*length*(all1[u] - 2*B)
                      + W*length*length

    point = locate_on_original_tree(best, index)
    verify_directional_optimality(point, V)
    mse = compensated_sum(a[j] * distance(point, x[j])**2) / W
    return point, mse
```

### Numerical implementation guidance

The equations above are exact algebraically, but rerooted second moments and minimized quadratic values can subtract nearly equal large numbers. Do not select a materially wrong point because two cancellation-corrupted objective values look equal. Recomputing the final objective alone cannot repair a wrong selection.

A preferable production selection rule uses first-moment directional derivatives, exploiting global convexity. At vertex u, let B_dir be the first moment in an incident direction, measured from u. The derivative into that direction is 2(H[u]-2B_dir). A vertex is optimal exactly when all incident derivatives are nonnegative. An edge contains the interior optimum exactly when t=(2B-H[u])/W lies strictly between 0 and ell. The interior zero derivative or a vertex satisfying all directions identifies the global minimizer; second moments are then only needed for reporting or cross-checks.

To avoid subtracting a large child contribution from a nearly equal total, compute complementary first moments with nonnegative message sums. Store up-mass and up-first-moment for each node, initialized to zero at the root. For child v of u, let C be mass at u plus up-mass[u] plus sibling subtree masses, and let E be up-first[u] plus sibling contributions (h_z + length_z*m_z). Then up-mass[v]=C and up-first[v]=E+ell*C. Prefix/suffix sums over children compute all sibling exclusions in linear total time. At u, the edge's two first moments are B and E, so t=(B-E)/W. The parent-side directional moment at a vertex is its up-first moment.

Use scale-aware tolerances for derivative zero and endpoints. Contract true zero-length edges before these checks. Accumulate many small contributions carefully, and recompute sensitive cases in long double. A tiny negative MSE caused by arithmetic may be clamped only after a scale-aware consistency check; a large negative value is a failed computation. Preserve full precision in serialization rather than rounding to four decimals.

## Mapping the winner to the original jplace edge

If the winner is an existing virtual vertex, its location descriptor already identifies the original point. Otherwise it is an offset s along a virtual edge whose upper endpoint u is ancestral to its lower endpoint v. Its root distance is D_target = D(u)+s.

Use binary ancestors, starting from the lower endpoint's original descendant anchor, to find the original edge whose endpoint root distances bracket D_target. For a lower endpoint inside an edge, its original child is a valid starting anchor. Ascend while an ancestor remains strictly deeper than the target, then bracket with the parent. Handle exact vertex equality explicitly. On the contracted positive-length tree this costs O(log n). Use extended precision or accumulated lengths for comparisons when absolute root distances dwarf local lengths.

For the winning original edge with child z,

\[
 \texttt{distal\_length}=D(z)-D_{\rm target}.
\]

Use one documented canonical representation at vertices, for example the smallest original incident edge ID; distal is zero at its distal endpoint or the edge length at its proximal endpoint. Precompute the canonical representation for zero-length equivalence classes. This convention affects serialization only, not the geometric answer. Do not introduce an epsilon displacement away from vertices.

## Output fields and diagnostics

The core result must include original `edge_num`, exact `distal_length`, group identity, query count, and mean squared distance. Root mean squared distance is sqrt(MSE), in branch-length units. Optional mean distance can preserve a familiar error diagnostic, but it is not the optimized score. Do not put MSE into a column previously documented as mean distance or edge-count error.

For jplace output, emit a single placement per consensus with `like_weight_ratio=1`. The objective does not estimate pendant length or a sequence likelihood. Proposed serialization policy: use the same equal-query mixture to average input pendant lengths if all are available, otherwise use an explicitly documented zero placeholder; mark this as a reporting convention in metadata. Do not include pendant length in optimization. If a `likelihood` column is retained for compatibility, use a documented placeholder (such as the existing zero convention), not `-MSE` presented as sequence likelihood.

Produce an explicit output field schema for the new method, typically the five standard fields, and remove unsupported input-specific placement columns rather than copying values from an arbitrary input row or silently filling unknown statistics. Preserve the original tree and nonconflicting top-level metadata, and add method, weighting, coordinate, and pendant/likelihood conventions to metadata. Return richer diagnostics separately in the Python result or a sidecar TSV. Adapt existing return shapes at the wrapper boundary so callers can choose the new method without losing group associations.

Any downstream routine that needs a consensus position must consume both edge ID and distal length. A node label alone discards information essential to the new method.

## Parallelism and batching

Parse and validate input, resolve groups, and build the tree index before launching workers. Parallelize over independent consensus groups using OpenMP, with thread-local virtual trees and moment buffers. Avoid nested OpenMP loops. Write results into preallocated slots in deterministic group order and serialize files after computation. Do not write shared streams or mutate grouping maps inside numerical worker loops.

Convert C++ exceptions to Python exceptions in the new SWIG boundary. Inside parallel loops, capture per-group failures safely and report them after the parallel region; do not let exceptions escape an OpenMP worker. Handle zero groups before computing thread counts, and require a positive requested thread count.

For large inputs, expose batching so placement buffers can be released while the reference index remains alive. Actual runtime will depend on group-size distribution; benchmark serial and parallel paths rather than assuming more threads always help.

## Validation and acceptance criteria

### Analytic and semantic cases

1. One point: return exactly its original geometric point with MSE zero.
2. Two equal weights: return the path midpoint, even if it lies on an original edge with no input placement.
3. Two weights a and b: distance from the first point is b*d(x1,x2)/(a+b).
4. Several points on one edge or a line: match the ordinary weighted arithmetic mean in linear coordinates.
5. Three equal masses at unit-length arms of a star: return the center with MSE one.
6. Unequal star: put weights 0.8, 0.1, 0.1 at the three unit tips; return distance 0.6 from the center toward the heavy tip.
7. Equal-query weighting: the example in the objective section must return 5. Multiply all raw weights of one query by a positive constant; the result must not change after within-query normalization.
8. Split one placement into coincident rows whose weights sum to the original weight; the result must not change. More placement rows must not increase a query's total influence.
9. Pool distributions rather than query means: query A has half mass at each of two unit tips of a three-arm star; query B has all mass at the third tip. The pooled mean is the star center. The mean of the two query means is halfway along the third arm and is wrong.
10. Roots, leaves, internal vertices, same-edge points, multiple placements at one vertex through different incident edges, zero-length components, and positive root stems must obey the declared coordinate policy.

### Independent geometric oracle

For small random trees, explicitly split every occupied original edge at observation points. Enumerate every resulting original segment. Independently compute direct distances to all observations, form its one-dimensional quadratic, and evaluate its clipped analytic minimum. Compare this exhaustive result to the virtual-tree solver in objective and geometric distance, not only serialized edge labels.

Use deterministic random seeds and include long chains, combs, multifurcations, tiny/large length mixtures, endpoint placements, zero weights, and duplicate positions. Test edge subdivision invariance and root-change invariance of the geometric result. A dense grid alone is not an adequate oracle for exact edge-interior minimization.

As a check on this specification, a temporary standalone Python calculation tested the postorder moments, reroot formulas, and compressed-edge minimization against an independent exhaustive original-edge oracle on 2,000 randomly generated weighted trees. All comparisons passed. That check treated observations as vertices after conceptual subdivision; it did not validate a JEST parser, SWIG interface, production precision policy, or final edge lookup. Production tests must cover those separately.

### JEST integration tests

- Permute `fields`; verify identical results and correctly ordered output rows.
- Use nonconsecutive edge IDs, quoted/awkward labels supported by the parser, and labels that are substrings of others.
- Test positions/lengths smaller than 1e-4 so old output rounding cannot pass unnoticed.
- Test integer edge IDs beyond float's exact-integer range to catch accidental float conversion.
- Test `n`, multi-name records, `nm`, absent mapping, file mapping, regex mapping, duplicate names, and malformed inputs under the declared policy.
- Verify disabled filtering really leaves all valid queries in the new method's groups; verify retained groups are renormalized after filtering.
- Verify unsupported multi-copy/feature-weight combinations fail explicitly.
- Round-trip jplace geometry, preserve tree annotations, and validate 0 <= distal <= edge length.
- Confirm serial and multithreaded results agree within the numerical policy and maintain stable group order.
- Keep regression tests for the legacy method, including wrapper return shapes.
- Rebuild the SWIG extension and exercise the installed CLI/API, not just a standalone C++ helper. `setup.py` currently packages an extension artifact rather than defining a normal extension build, so explicitly verify that the tested binary includes the new code.

### Performance tests

Benchmark n around 10,000, 100,000, and several hundred thousand, with small and larger k, clustered placements, and placements separated by long original paths. Separate index construction, virtual-tree construction, solving, final edge lookup, and serialization. Record peak memory and throughput for representative group sizes. Holding k fixed while increasing the number of intervening edges must not cause a linear increase in solve time. No full-tree copy, ancestor walk, Newick search, or n-sized scratch reset may appear inside the per-group hot path.

## Suggested implementation sequence

1. Implement typed parsing, logical-query grouping, validation, and double-precision tree indexing; test coordinates and original ID preservation.
2. Implement an exhaustive small-tree reference solver and analytic cases before introducing compression.
3. Implement virtual-tree construction and moment solving; compare against the independent oracle.
4. Implement stable derivative-based candidate selection, original-edge localization, and full-precision output.
5. Add SWIG/API/CLI method routing, equal-query pooling, explicit filtering behavior, and metadata conventions.
6. Validate grouping, serialization, precision, zero-length behavior, parallel execution, and legacy compatibility.
7. Benchmark the shared-index batch path and document measured behavior and the precise complexity bound.

The Frechet deliverable is a deterministic continuous tree Frechet consensus method with equal total contribution per query. No kernel bandwidth, density normalization, midpoint approximation, or nonlinear global optimizer belongs in the Frechet solver. The optional exponential-kernel solver below has its own decay parameter.

## Exponential kernel mode extension

### Objective and behavior

Add an optional `exponential_mode` method that maximizes

\[
 S(c)=\frac{1}{W}\sum_j a_j\exp(-\alpha d(c,x_j)),
 \qquad W=\sum_j a_j,\quad \alpha>0.
\]

Use the same equal-query pooling as Frechet consensus: independently normalize each query's placement distribution, then combine queries with equal total contribution. The Python implementation currently omits the common 1/q factor when passing pooled weights to C++; consequently divide the final kernel sum by the actual W, not by row count. Positive rescaling of all pooled weights must leave the answer and normalized score unchanged.

This maximizes expected exponential proximity, or equivalently minimizes expected loss 1-exp(-alpha*d). It is not a normalized probability density on the tree and is not the maximum-likelihood center of independent observations under exponential radial decay. Do not introduce Z(c) or any topology-dependent normalization.

Alpha has inverse branch-length units. The characteristic distance 1/alpha gives attenuation exp(-1). Require an explicit finite positive `kernel_alpha` for this method initially; do not invent a data-dependent default. In the CLI use `--kernel-alpha`. JEST already uses `alpha` for filtering, so those parameters must remain separate. Reject a supplied kernel parameter for other methods rather than silently ignoring it. Reject zero: at alpha=0 every point ties and the candidate reduction below does not describe all optimizers.

Unlike the Frechet mean, this objective can have multiple global maxima. For two distinct points, an endpoint wins; equal weights tie, and unequal weights select the heavier endpoint. For three equal-weight placements on equal-length star arms, a branching vertex can win. This behavior is part of the method, not a numerical failure.

### Exact candidate reduction

Let H be the smallest subtree connecting the positive-weight placements. For a point c outside H with projection p onto H,

\[
 S(c)=e^{-\alpha d(c,p)}S(p)\le S(p).
\]

Therefore a maximum occurs in H. Split H at placement points and its branching vertices, exactly as in the virtual-tree construction. On a remaining segment of length ell, with coordinate s from its first endpoint, the unnormalized score has the form

\[
 S_{\rm raw}(s)=A e^{-\alpha s}+B e^{-\alpha(\ell-s)},\qquad A,B\ge0.
\]

There are no placements or paths to placements branching off that segment's interior. Thus

\[
 S_{\rm raw}''(s)=\alpha^2 S_{\rm raw}(s)>0.
\]

Its maximum is at an endpoint. All necessary candidates are placement points and retained original branching vertices, O(k) in total. Retaining the extra original endpoints used by the existing virtual-tree builder is harmless: these are valid tree points and cannot beat the true global maximum. Degree-two original vertices suppressed on compressed paths need not be evaluated. The same argument covers arbitrary distances between the observations; proximity is not an assumption.

### Two message passes

Root the virtual tree as in the Frechet implementation. Let a[v] be mass exactly at vertex v and let r[u,v]=exp(-alpha*ell[u,v]). Compute this attenuation once per virtual edge using its full compressed length.

Postorder computes the contribution of observations in a vertex's descendant subtree:

\[
 D[v]=a[v]+\sum_{z\text{ child of }v}r[v,z]D[z].
\]

Preorder computes the contribution of all other observations, initializing U[root]=0:

\[
 U[z]=r[v,z]\left(U[v]+a[v]+\sum_{b\ne z}r[v,b]D[b]\right).
\]

Then the raw score at v is D[v]+U[v]. Evaluate every virtual vertex and select the largest score. Compute sibling-exclusion sums with prefix/suffix sums so the pass is O(k) even at high-degree vertices, and so it never subtracts nearly equal nonnegative totals.

```text
for v in postorder:
    down[v] = own_mass[v]
    for child in children[v]:
        attenuation[child] = exp(-kernel_alpha * edge_length(v, child))
        down[v] += attenuation[child] * down[child]

up[root] = 0
for v in preorder:
    contributions = [attenuation[z] * down[z] for z in children[v]]
    prefix, suffix = nonnegative_prefix_suffix_sums(contributions)
    score[v] = down[v] + up[v]
    for i, z in enumerate(children[v]):
        outside = up[v] + own_mass[v] + prefix[i] + suffix[i+1]
        up[z] = attenuation[z] * outside

winner = argmax_with_deterministic_ties(score)
output_point = serialize_existing_virtual_location(winner)
kernel_score = direct_weighted_kernel_sum(output_point, original_points) / W
```

Prefix[i] sums indices strictly before i and suffix[i+1] sums indices strictly after i. Preserve `own_mass` before any subtree accumulation; the existing Frechet `mass` field is overwritten with subtree mass and cannot be used as a[v] afterwards.

Do not reuse the Frechet descent based on a unique negative derivative or its quadratic edge minimizer. Exponential mode requires scanning all candidate vertex scores; it does not have Frechet's uniqueness or global convex minimization property.

### Integration with the implemented code

The inspected checkout has `FrechetTreeIndex::Impl::solve` in `JEST/cbinder/frechet_consensus.cpp`, `FrechetTreeIndex::solve_batch` in its header, and pooling/serialization in `JEST/frechet.py`. The virtual tree is presently constructed inside `Impl::solve`. Extract its construction, location descriptors, traversal order, and original own masses into an internal reusable helper. Keep the immutable parser/index and its coordinate validation unchanged. Preserve any median implementation added on another branch; do not replace or narrow an existing method selector to the methods listed here.

Add a new low-level entry point such as `exponential_mode_batch(..., kernel_alpha, threads, profile=false)` with the same numeric buffers, group offsets, query counts, and exception handling as the existing batch method. A separate entry point avoids changing the positional signature or default behavior of `solve_batch`. Add its declaration and result conversion in `JEST/cbinder/CTree.i`, regenerate SWIG artifacts, and rebuild the extension.

Use a dedicated result struct, or a deliberately generalized shared result type, with a `kernel_score` field. Do not repurpose `mean_squared_distance` to store the optimized score. Optionally compute ordinary mean distance and MSE at the winning point for comparison, labeling them as diagnostics rather than optimized objectives. They require only O(k) extra indexed distance evaluations.

Generalize the Python dispatch while retaining its normalization, logical-query identities, multi-name handling, metadata conventions, and pendant-length reporting. Add `method="exponential_mode"` to both placement and grouped consensus entry points. Preserve existing return shapes and rich-diagnostics behavior. Metadata must identify the method, kernel_alpha, equal-query weighting, score definition, and tie policy; do not label exponential results as Frechet. Keep likelihood a documented placeholder rather than treating the kernel score as sequence likelihood.

The new method does not automatically inherit Frechet filtering calibration. Existing filtering can be retained as an explicitly documented independent Frechet-based preprocessing option, or disabled for this method initially. Any filtering that substitutes an exponential-mode center or kernel score requires matching null calibration. In particular, do not silently reuse Frechet null scores after changing the statistic. Reject unsupported filter combinations explicitly until their semantics and tests are implemented.

### Numerical details and ties

Use the existing extended-precision accumulator type for messages and sums where available. Compute alpha*ell without an avoidable intermediate double overflow. Attenuation underflow to zero for extremely large alpha*ell represents a contribution below working precision; never discard a placement's self-contribution. Every group with positive mass has a strictly positive winning raw score.

For extremely small alpha, all normalized scores approach one and differences may be below working precision. Use compensated sums and extended precision, and test small-alpha cases. If distinguishability below that precision is required, use higher precision or a separately validated small-alpha expansion; do not promise exact ordering for arbitrarily close scores. The leading small-alpha term favors tree medians, but is not generally an exact substitute for the finite-alpha objective.

Select the numerical maximum first. For scores equal within a declared tolerance relative to the group mass, report a numerical tie and choose deterministically, for example using original vertex identity or (original edge ID, distal coordinate) for interior placements. Apply any tolerance relative to the maximum, not as a chain of pairwise near-equality comparisons. A tolerance tie is not proof of an exact mathematical tie. Do not average tied locations: the midpoint of two modes can have a lower score.

Original vertices already have canonical serialized endpoints, and every interior candidate is an observed coordinate. Preserve that provenance during virtual-tree construction to serialize the winner directly, avoiding the weighted-ancestor search required for a Frechet interior optimum. Do not locate every candidate on an original edge merely to break ties. If reusing the existing generic `locate` routine, call it once after selection and retain the extra O(log n) term in the advertised bound.

Recompute the selected score at the serialized coordinate with direct indexed point distances as an O(k) consistency check. The normalized result must lie in (0,1] up to roundoff. A material disagreement with the message score is an implementation error. A group supported at a single geometric point has score one.

### Complexity

Reuse O(n log n) shared preprocessing. A group takes O(k log(k+1)) to construct its virtual tree, O(k) for both message passes and candidate selection, and O(k) for direct verification, with O(k) working memory. Saving candidate provenance gives O(1) final serialization and the full O(k log(k+1)) bound. Reusing generic final localization instead gives O(k log(k+1)+log n). There are O(k) exponential evaluations per pass/check, no O(k squared) production scoring, and no per-query traversal of original intervening edges. Parallelize over groups using the existing immutable index and thread-local buffers.

### Tests and acceptance criteria

1. One geometric point, including duplicates: return that point with normalized score one.
2. Two points separated by L: endpoint scores are (a+b*exp(-alpha*L))/W and (b+a*exp(-alpha*L))/W. The heavier endpoint wins; equal masses tie. Verify the midpoint scores lower for positive L and alpha.
3. Three equally weighted unit star tips: put r=exp(-alpha). The center scores r and each tip scores (1+2*r*r)/3. The center wins for 1/2<r<1, tips win for 0<r<1/2, and all four candidates tie at r=1/2. Test all regimes and deterministic ties.
4. Multiple distinct placements on one edge: include every observed coordinate as a candidate and compare directly evaluated scores. Never collapse the whole edge's mass to an endpoint.
5. Equal-query pooling: change the number of rows by splitting weights among duplicate coordinates, or rescale one query's raw weights before normalization. The normalized answer and score must remain unchanged.
6. Tiny/large kernel_alpha, long compressed paths, high-degree vertices, zero-length components, true vertex ties, near ties, positive root stems, and nonuniform branch lengths must follow the stated precision and coordinate policies.
7. Build an independent small-tree oracle by subdividing the full original tree at every observed point and directly scoring every resulting vertex. The convexity proof makes this a complete finite candidate oracle; it must not reuse production message passes. Compare maximum score and returned geometric location/tie membership on seeded random trees. A dense grid may supplement but must not replace this oracle.
8. Re-root and subdivide unoccupied edges without changing geometry. The maximum score and unique winning point must be unchanged. Check serial/parallel agreement, field-order independence, metadata, parameter validation, and both API/CLI routes.
9. Benchmark fixed k with increasingly long original paths. Solve time must not grow linearly with original-tree size. Include repeated calls with different kernel_alpha values using the same tree index.

This extension adds an exact-in-real-arithmetic exponential proximity mode, not the Gaussian-kernel mode. Squaring distance inside the exponential invalidates the endpoint convexity argument and requires a different optimizer.
