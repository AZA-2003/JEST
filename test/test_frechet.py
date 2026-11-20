"""Tests exercise the compiled SWIG extension, not a substitute Python solver."""
from copy import deepcopy
import json
import math
import random

import pytest
from typer.testing import CliRunner

from JEST import placement_consensus, subgenome_consensus
from JEST.JEST import app
from JEST.CTree import FrechetTreeIndex
from JEST.frechet import consensus

FIELDS = ["edge_num", "like_weight_ratio", "distal_length", "pendant_length"]


def doc(tree, queries):
    return {"tree": tree, "fields": FIELDS.copy(), "placements": [
        {"n": [f"q{i}"], "p": [[e, w, d, 2.0] for e, w, d in rows]}
        for i, rows in enumerate(queries)], "metadata": {"source": "test"}, "version": 3}


def solve(tree, queries, grouped=True, **kwargs):
    out, diagnostics = consensus(doc(tree, queries), grouped=grouped, **kwargs)
    return out["placements"][0]["p"][0], diagnostics[0]


def test_line_equal_query_normalization():
    tree = "(a:12{100})root:0{9};"
    for factor in [1, 1e200, 1e-200]:
        row, diag = solve(tree, [[(100, factor, 12)], [(100, 1, 4), (100, 1, 0)]])
        assert row == pytest.approx([100, 0, 1, 7, 2])
        assert diag["query_count"] == 2
        assert diag["mean_squared_distance"] == pytest.approx(27)
    row, _ = solve(tree, [[(100, 1, 12), (100, 1, 4), (100, 1, 0)]])
    assert row[3] == pytest.approx(12 - 20 / 3)


@pytest.mark.parametrize("rows,edge,distal,mse", [
    ([(20, 1, 0), (30, 1, 0), (40, 1, 0)], 20, 1, 1),
    ([(20, .8, 0), (30, .1, 0), (40, .1, 0)], 20, .4, .64),
    ([(20, 1, .123456789123)], 20, .123456789123, 0),
])
def test_analytic_star(rows, edge, distal, mse):
    row, diag = solve("(a:1{20},b:1{30},c:1{40});", [rows])
    assert row[0] == edge
    assert row[3] == pytest.approx(distal, abs=1e-15)
    assert diag["mean_squared_distance"] == pytest.approx(mse, abs=1e-28)


def test_pool_distributions_not_query_centers():
    row, diag = solve("(a:1{20},b:1{30},c:1{40});",
                      [[(20, 1, 0), (30, 1, 0)], [(40, 1, 0)]])
    assert row[:4] == [20, 0, 1, 1]
    assert diag["mean_squared_distance"] == 1


def test_zero_edges_and_root_stem():
    row, diag = solve("((a:0{7},b:0{8}):0{4},c:0{9}):0{2};",
                      [[(7, 1, 0), (9, 2, 0)]])
    assert row[:4] == [2, 0, 1, 0]
    assert diag["mean_squared_distance"] == 0
    row, diag = solve("(a:2{7}):4{3};", [[(3, 1, 4), (7, 1, 0)]])
    assert row[:4] == [3, 0, 1, 1]
    assert diag["mean_squared_distance"] == 9
    row, _ = solve("((a:1{9}):0{2},b:1{7});", [[(9, 1, 0), (7, 1, 0)]])
    assert row[:4] == [2, 0, 1, 0]


def test_precision_fields_names_and_metadata():
    tree = "('a:b,{x}':0.00000007{18446744073709551615},'a''b':1{99}):0{5};"
    document = doc(tree, [[(2**64-1, 1, 0.0000000123456789)]])
    document["placements"][0].pop("n")
    document["placements"][0]["nm"] = [["one", 9], ["two", 3]]
    document["fields"] = ["pendant_length", "distal_length", "like_weight_ratio", "edge_num", "custom"]
    document["placements"][0]["p"] = [[2, .0000000123456789, 1, 2**64-1, 555]]
    out, diags = consensus(document)
    assert out["tree"] == tree
    assert out["placements"][0]["nm"] == document["placements"][0]["nm"]
    assert out["placements"][0]["p"][0][0] == 2**64-1
    assert out["placements"][0]["p"][0][3] == pytest.approx(.0000000123456789, rel=1e-15)
    assert out["metadata"]["source"] == "test"
    assert diags[0]["query_count"] == 2
    assert len(out["fields"]) == 5
    second, _ = consensus(out)
    assert second["placements"] == out["placements"]


def test_grouping_and_parallel(tmp_path):
    document = doc("(a:12{100});", [[(100, 1, 12)], [(100, 1, 0)]])
    document["placements"][0]["n"] = ["A_one", "B_one"]
    document["placements"][1]["nm"] = [["A_two", 100], ["B_two", 1]]
    del document["placements"][1]["n"]
    options = dict(grouped=True, genome_map=r"(.*)_(.*)")
    a = consensus(document, **options, num_threads=1)
    b = consensus(document, **options, num_threads=4)
    assert a == b
    assert [p["n"] for p in a[0]["placements"]] == [["A"], ["B"]]
    assert all(p["p"][0][3] == 6 for p in a[0]["placements"])
    mapping = tmp_path / "map.tsv"
    mapping.write_text("A_one\tA\nB_one\tB\nA_two\tA\nB_two\tB\n")
    assert consensus(document, grouped=True, genome_map=str(mapping)) == a
    mapping.write_text("A_one\tA\t0.5\n")
    with pytest.raises(ValueError, match="Feature-weighted"):
        consensus(document, grouped=True, genome_map=str(mapping))
    for content in ["A_one\tA\n", "A_one\tA\nA_one\tB\n", "broken\n"]:
        mapping.write_text(content)
        with pytest.raises(ValueError):
            consensus(document, grouped=True, genome_map=str(mapping))
    for pattern in [".*", "(Z)_(.*)", "("]:
        with pytest.raises(ValueError):
            consensus(document, grouped=True, genome_map=pattern)


@pytest.mark.parametrize("change", [
    lambda d: d["fields"].remove("distal_length"),
    lambda d: d["placements"][0]["p"][0].__setitem__(0, 1.2),
    lambda d: d["placements"][0]["p"][0].__setitem__(0, 999),
    lambda d: d["placements"][0]["p"][0].__setitem__(1, -1),
    lambda d: d["placements"][0]["p"][0].__setitem__(1, float("nan")),
    lambda d: d["placements"][0]["p"][0].__setitem__(1, 0),
    lambda d: d["placements"][0]["p"][0].__setitem__(2, -1e-5),
    lambda d: d["placements"].append(deepcopy(d["placements"][0])),
    lambda d: d["placements"][0].__setitem__("n", ["q", "q"]),
    lambda d: d.__setitem__("placements", []),
    lambda d: d.__setitem__("tree", "(a:-1{20});"),
    lambda d: d.__setitem__("tree", "(a:1{20},b:2{20});"),
    lambda d: d.__setitem__("tree", "(a:1{20}):2;"),
    lambda d: d.__setitem__("tree", "(a:1{20},);"),
])
def test_invalid_inputs(change):
    document = doc("(a:1{20});", [[(20, 1, .5)]])
    change(document)
    with pytest.raises((ValueError, TypeError)):
        consensus(document)


def test_api_and_cli(tmp_path, monkeypatch):
    document = doc("(a:12{100});", [[(100, 1, 12)], [(100, 1, 0)]])
    path = tmp_path / "input.jplace"
    path.write_text(json.dumps(document))
    # Disabled filtering must not call any random/legacy uncertainty routines.
    from JEST.JEST import CTree
    def forbidden(*args):
        raise AssertionError("Legacy code called")
    monkeypatch.setattr(CTree, "get_random_placements_uncertainty", forbidden)
    monkeypatch.setattr(CTree, "get_uncertainty_pvalue", forbidden)
    result, remaining = subgenome_consensus(str(path), True, str(tmp_path), method="frechet")
    assert len(remaining) == 2 and result[0]["p"][0][3] == 6
    detailed = placement_consensus(str(path), False, method="frechet", return_diagnostics=True)
    assert len(detailed["diagnostics"]) == 2
    for kwargs in [{"filter_queries": True}, {"multi_copy": True}, {"edge_error": True}, {"num_threads": 0}]:
        with pytest.raises(ValueError):
            subgenome_consensus(str(path), False, method="frechet", **kwargs)
    for method in ["oops", "Frechet"]:
        with pytest.raises(ValueError):
            placement_consensus(str(path), False, method=method)
    cli = CliRunner().invoke(app, ["subgenome-consensus", "-i", str(path), "-w", "-o", str(tmp_path), "--method", "frechet", "-e"])
    assert cli.exit_code == 0, cli.output + str(cli.exception)
    saved = json.loads((tmp_path / "input_subgenome_consensus.jplace").read_text())
    assert saved["tree"] == document["tree"]
    assert saved["metadata"]["frechet"]["diagnostics"][0]["query_count"] == 2
    assert (tmp_path / "input_subgenome_frechet.tsv").exists()


def test_legacy_wrapper_routing(tmp_path, monkeypatch):
    from JEST.JEST import CTree
    document = doc("(a:1{0},b:1{1}):0{2};", [[(0, 1, .2)]])
    path = tmp_path / "legacy.jplace"
    path.write_text(json.dumps(document))
    called = []
    def legacy(*args):
        called.append(args)
        return document["placements"]
    monkeypatch.setattr(CTree, "placement_consensus", legacy)
    assert placement_consensus(str(path), False) == document["placements"]
    assert len(called) == 1


def oracle(parent, lengths, points):
    """Split occupied edges, enumerate all original segments, use graph distances.

    This does not use root depths, LCA, virtual trees, or message equations.
    """
    n = len(parent)
    adjacency = [[] for _ in parent]
    positions = {i: [] for i in range(1, n)}
    for e, distal, weight in points:
        positions[e].append((lengths[e] - distal, weight))
    masses = [0.0] * n
    segments = []
    for e in range(1, n):
        chain = [(0, parent[e]), (lengths[e], e)]
        for t, weight in positions[e]:
            if t == 0:
                masses[parent[e]] += weight
            elif t == lengths[e]:
                masses[e] += weight
            else:
                v = len(adjacency)
                adjacency.append([])
                masses.append(weight)
                chain.append((t, v))
        chain.sort()
        for (t0, a), (t1, b) in zip(chain, chain[1:]):
            adjacency[a].append((b, t1-t0))
            adjacency[b].append((a, t1-t0))
            segments.append((a, b, t1-t0, e, t0))
    def distances(start, blocked=None):
        ds = {start: 0.0}
        stack = [start]
        while stack:
            u = stack.pop()
            for v, length in adjacency[u]:
                if v not in ds and v != blocked:
                    ds[v] = ds[u] + length
                    stack.append(v)
        return ds
    W = sum(masses)
    best = None
    for a, b, length, e, t0 in segments:
        ds = distances(a)
        bside = distances(b, blocked=a)
        linear = math.fsum(w * ds[v] * (-1 if v in bside else 1) for v, w in enumerate(masses))
        s = min(length, max(0, -linear / W))
        objective = math.fsum(w * (ds[v] + (-s if v in bside else s))**2 for v, w in enumerate(masses)) / W
        if best is None or objective < best[0]:
            best = (objective, e, lengths[e] - (t0+s))
    # Independent geometry between any pair of original-edge coordinates.
    vertex_ds = [distances(i) for i in range(n)]
    def point_distance(e, d, f, z):
        if e == f:
            return abs(d-z)
        return min(d+vertex_ds[e][f]+z,
                   d+vertex_ds[e][parent[f]]+lengths[f]-z,
                   lengths[e]-d+vertex_ds[parent[e]][f]+z,
                   lengths[e]-d+vertex_ds[parent[e]][parent[f]]+lengths[f]-z)
    return best, point_distance


def make_newick(parent, lengths):
    children = [[] for _ in parent]
    for v in range(1, len(parent)):
        children[parent[v]].append(v)
    parts = {}
    for v in range(len(parent)-1, -1, -1):
        label = "(" + ",".join(parts[c] for c in children[v]) + ")" if children[v] else "x"
        parts[v] = label + (f":{lengths[v]:.17g}{{{v}}}" if v else ";")
    return parts[0]


def test_random_exhaustive_oracle():
    rng = random.Random(9821)
    for trial in range(2000):
        n = rng.randrange(2, 16)
        parent = [0] + [rng.randrange(v) if trial % 3 else v-1 for v in range(1, n)]
        lengths = [0] + [rng.choice([0, 1e-5, .1, 1, 100]) * rng.uniform(.5, 2) for _ in range(n-1)]
        points = []
        for _ in range(rng.randrange(1, 16)):
            e = rng.randrange(1, n)
            points.append((e, rng.choice([0, lengths[e], rng.random()*lengths[e]]), rng.random()))
        tree = make_newick(parent, lengths)
        index = FrechetTreeIndex(tree)
        result = index.solve_batch([p[0] for p in points], [p[1] for p in points], [p[2] for p in points], [0, len(points)], [1], 1)[0]
        best, distance = oracle(parent, lengths, points)
        scale = max(1, sum(lengths))
        assert result.mean_squared_distance == pytest.approx(best[0], rel=1e-10, abs=1e-15*scale**2), trial
        assert distance(result.edge_num, result.distal_length, best[1], best[2]) < 1e-9*scale, trial


def test_deep_comb_and_unobserved_edge():
    # Iterative parser/index; the midpoint must be found on an unoccupied edge.
    n = 10000
    tree = "(" * n + "x:1{1}" + "".join(f"):1{{{v}}}" for v in range(2, n+1)) + ");"
    index = FrechetTreeIndex(tree)
    result = index.solve_batch([1, n], [0, 1], [1, 1], [0, 2], [2], 1)[0]
    assert result.mean_squared_distance == pytest.approx((n/2)**2)
    assert result.edge_num not in [1, n]
    assert result.distal_length in [0, 1]


def test_batch_validation_and_reuse():
    index = FrechetTreeIndex("(x:1{7});")
    assert len(index.solve_batch([], [], [], [0], [], 2)) == 0
    for _ in range(3):
        assert index.solve_batch([7], [.25], [1], [0, 1], [1], 2)[0].distal_length == .25
    for offsets, counts in [([0, 0], [1]), ([0, 2, 1], [1, 1]), ([0, 1], [0])]:
        with pytest.raises(ValueError):
            index.solve_batch([7], [0], [1], offsets, counts, 1)


def test_subdivision_and_rerooting():
    original, diagnostic = solve("(a:10{7},b:3{8},c:2{9});",
                                 [[(7, .7, 0), (8, .2, 0), (9, .1, 0)]])
    # Subdivide the heavy arm 4 units from its tip, then reroot at that tip.
    subdivided, d2 = solve("((a:4{7}):6{17},b:3{8},c:2{9});",
                          [[(7, .7, 0), (8, .2, 0), (9, .1, 0)]])
    rerooted, d3 = solve("((b:3{8},c:2{9}):10{27})a;",
                        [[(27, .7, 10), (8, .2, 0), (9, .1, 0)]])
    assert original[3] == pytest.approx(3.8)
    assert subdivided[0] == 7 and subdivided[3] == pytest.approx(original[3])
    assert rerooted[3] == pytest.approx(10-original[3])
    assert d2["mean_squared_distance"] == pytest.approx(diagnostic["mean_squared_distance"])
    assert d3["mean_squared_distance"] == pytest.approx(diagnostic["mean_squared_distance"])


def test_split_weights_zero_weights_and_missing_pendants():
    document = doc("(a:1{1},b:1{2},c:1{3});", [[(1, .8, 0), (2, .1, 0), (3, .1, 0)]])
    original = consensus(document)
    document["placements"][0]["p"][0:1] = [[1, .2, 0, 2], [1, .6, 0, 2], [1, 0, .4, 2]]
    out, diagnostics = consensus(document)
    assert out["placements"][0]["p"][0] == pytest.approx(original[0]["placements"][0]["p"][0])
    assert diagnostics[0]["mean_squared_distance"] == pytest.approx(original[1][0]["mean_squared_distance"])
    document["fields"].pop()
    for row in document["placements"][0]["p"]:
        row.pop()
    out, _ = consensus(document)
    assert out["placements"][0]["p"][0][-1] == 0
    assert "zero placeholder" in out["metadata"]["frechet"]["pendant_length"]


def test_cancellation_resistant_selection():
    # Large second moments and a tiny local displacement; selecting by rounded
    # objective values would incorrectly tie many of the possible positions.
    row, diag = solve("(a:1000000000{1},b:1000000000{2},c:0.00001{3});",
                      [[(1, .5, 0), (2, .5-1e-12, 0), (3, 1e-12, 0)]])
    assert row[0] == 1
    assert row[3] == pytest.approx(1e9-.001, abs=2e-7)
    assert diag["mean_squared_distance"] > 1e17


def test_unrepresentable_coordinates_fail_instead_of_merging():
    with pytest.raises((ValueError, RuntimeError), match="precision"):
        solve("(x:1{1}):10000000000000000{2};",
              [[(1, 1, .5), (1, 1, .5000000000001)]])
