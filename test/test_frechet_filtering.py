"""Filtering tests use the rebuilt native index and independent distance checks."""
import csv
import json
import math
from copy import deepcopy

import pytest
from typer.testing import CliRunner
from JEST import subgenome_consensus
from JEST.JEST import app
from JEST.CTree import FrechetTreeIndex
from JEST.frechet import consensus
from JEST.frechet_filtering import filtered_consensus, _score


def document(rows, names=None):
    return {"tree": "(tip:10{9});", "fields": ["edge_num", "like_weight_ratio", "distal_length"],
            "placements": [{"n": [names[i] if names else f"q{i}"], "p": [[9, w, d] for d, w in row]}
                           for i, row in enumerate(rows)], "metadata": {"source": "test"}}


def filtered(doc, **kwargs):
    options = dict(genome_map="", num_threads=1, pre_filter=False, post_filter=False,
                   pre_alpha=.05, post_alpha=.05, trials=128, random_placements=10, random_seed=48)
    options.update(kwargs)
    return filtered_consensus(doc, **options)


def test_pre_filter_uses_own_center_then_equal_query_pooling():
    doc = document([[(0, 1)], [(10, 1)], [(0, 1), (10, 1)]])
    out, diagnostics, remaining, before, report = filtered(doc, pre_filter=True)
    assert [s["mean_distance"] for s in report["pre_scores"]] == [0, 0, 5]
    assert [s["reference_distal_length"] for s in report["pre_scores"]] == [0, 10, 5]
    assert report["pre_filtered_queries"] == ["q2"]
    assert [r["n"] for r in before] == [["q0"], ["q1"]]
    assert remaining == before
    assert diagnostics[0]["query_count"] == 2
    assert out["placements"][0]["p"][0][3] == 5
    expected, expected_diagnostics = consensus({**doc, "placements": before}, grouped=True)
    assert out["placements"] == expected["placements"]
    assert diagnostics == expected_diagnostics


def test_post_filter_is_point_aware_and_does_not_refit():
    doc = document([[(0, 1)], [(0, 1)], [(0, 1)], [(10, 1)]])
    out, diagnostics, remaining, before, report = filtered(doc, post_filter=True, post_alpha=.5)
    assert before == doc["placements"]
    assert out["placements"][0]["p"][0][3] == 2.5
    assert [s["mean_distance"] for s in report["post_scores"]] == [2.5, 2.5, 2.5, 7.5]
    assert report["post_filtered_queries"] == ["q3"]
    assert len(remaining) == 3
    assert diagnostics[0]["query_count"] == 4  # consensus still uses all pre-retained queries
    refit, _ = consensus({**doc, "placements": remaining}, grouped=True)
    assert refit["placements"] != out["placements"]
    assert report["pre_scores"] == []


def test_both_stages_renormalize_and_preserve_partial_nm():
    doc = document([[(0, 1)], [(0, 1)], [(0, 1)], [(10, 1)], [(0, 1), (10, 1)]],
                   ["A_1", "A_2", "A_3", "A_out", "A_diffuse"])
    doc["placements"][3] = {"nm": [["A_out", 100], ["B_good", 2]], "p": doc["placements"][3]["p"]}
    out, diagnostics, remaining, before, report = filtered(doc, pre_filter=True, post_filter=True,
                                                         genome_map=r"(.*)_(.*)", post_alpha=.5)
    assert report["pre_filtered_queries"] == ["A_diffuse"]
    assert report["post_filtered_queries"] == ["A_out"]
    assert before[3]["nm"] == [["A_out", 100], ["B_good", 2]]
    assert remaining[3]["nm"] == [["B_good", 2]]
    assert [d["query_count"] for d in diagnostics] == [4, 1]
    assert [p["n"] for p in out["placements"]] == [["A"], ["B"]]
    assert doc["placements"][3]["nm"][0][0] == "A_out"  # no mutation


def test_null_sampling_and_distance_primitives():
    index = FrechetTreeIndex("(a:1{1},b:3{2}):2{3};")
    assert list(index.random_points(0, 42)) == []
    with pytest.raises((OverflowError, TypeError)):
        index.random_points(-1, 42)
    a = [(p.edge_num, p.distal_length) for p in index.random_points(10000, 42)]
    b = [(p.edge_num, p.distal_length) for p in index.random_points(10000, 42)]
    assert a == b
    for edge, length in [(1, 1), (2, 3), (3, 2)]:
        selected = [d for e, d in a if e == edge]
        assert len(selected) / len(a) == pytest.approx(length / 6, abs=.02)
        assert all(0 <= d <= length for d in selected)
    # a's tip to b's tip is 4; a's tip to synthetic proximal root is 3.
    scores = index.mean_distance_batch([2, 3], [0, 2], [.25, .75], [0, 2], [1], [0], 4)
    assert list(scores) == [3.25]
    assert list(index.mean_distance_batch([], [], [], [0], [], [], 4)) == []
    for args in [([1], [0], [1], [0, 1], [1], [2], 1),
                 ([1], [0], [1], [0, 0], [1], [0], 1),
                 ([1], [0], [-1], [0, 1], [1], [0], 1)]:
        with pytest.raises(ValueError):
            index.mean_distance_batch(*args)


def test_seed_thread_reproducibility_and_stage_specific_calibration():
    doc = document([[(0, 1)], [(0, 1)], [(10, 1)], [(3, 1), (7, 1)]])
    first = filtered(doc, pre_filter=True, post_filter=True, pre_alpha=1, post_alpha=1)
    parallel = filtered(doc, pre_filter=True, post_filter=True, pre_alpha=1, post_alpha=1, num_threads=4)
    assert first == parallel
    report = first[-1]
    # Independently reconstruct each stage's null distribution.
    index = FrechetTreeIndex(doc["tree"])
    samples = list(index.random_points(1280, 48))
    pre_null, post_null = [], []
    center = first[1][0]["distal_length"]
    for i in range(128):
        ds = [p.distal_length for p in samples[i*10:(i+1)*10]]
        own = math.fsum(ds) / 10
        pre_null.append(math.fsum(abs(d-own) for d in ds) / 10)
        post_null.append(math.fsum(abs(d-center) for d in ds) / 10)
    for stage, null in [("pre", pre_null), ("post", post_null)]:
        for score in report[f"{stage}_scores"]:
            expected = sum(x < score["mean_distance"] for x in null) / len(null)
            assert score["null_percentile"] == expected
    assert pre_null != post_null


def test_percentile_direction_and_ties():
    ref = {"edge_num": 9, "distal_length": 1}
    assert _score("q", "g", 0, [0, 0, 0], 0, ref)["filtered"] is False
    assert _score("q", "g", 2, [1, 2, 3, 4], .25, ref)["filtered"] is False
    assert _score("q", "g", 5, [1, 2, 3, 4], .25, ref)["filtered"] is True


def test_zero_tree_and_empty_groups(tmp_path):
    doc = document([[(0, 1)]])
    doc["tree"] = "(tip:0{9});"
    out, diag, remaining, before, report = filtered(doc, pre_filter=True, post_filter=True,
                                                  pre_alpha=0, post_alpha=0)
    assert len(remaining) == 1
    assert report["pre_scores"][0]["mean_distance"] == 0
    assert report["post_scores"][0]["null_percentile"] == 0
    doc = document([[(0, 1), (10, 1)]])
    path = tmp_path / "empty.jplace"
    path.write_text(json.dumps(doc))
    result = subgenome_consensus(str(path), True, str(tmp_path), method="frechet",
                                pre_filter=True, post_filter=True, trials=32, get_error=True,
                                return_diagnostics=True)
    assert result["placements"] == []
    assert result["remaining_placements"] == []
    assert result["filtering"]["omitted_groups"] == ["consensus"]
    assert result["filtering"]["post_scores"] == []
    for filename in ["empty_subgenome_consensus.tsv", "empty_subgenome_frechet.tsv", "empty_postconsensus_filter.tsv"]:
        assert len((tmp_path / filename).read_text().splitlines()) == 1
    assert json.loads((tmp_path / "empty_preconsensus_filtered.jplace").read_text())["placements"] == []


@pytest.mark.parametrize("options", [
    {"pre_alpha": -1}, {"pre_alpha": float("nan")}, {"post_alpha": 2},
    {"trials": 0}, {"random_placements": 1}, {"random_seed": -1},
    {"random_seed": 2**64}, {"random_dist_path": "legacy.txt"},
])
def test_filter_option_validation(options):
    with pytest.raises(ValueError):
        filtered(document([[(0, 1)]]), pre_filter=True, post_filter=True, **options)


def test_disabled_filters_and_alias_cli_outputs(tmp_path, monkeypatch):
    doc = document([[(0, 1)], [(0, 1)], [(0, 1)], [(10, 1)]])
    path = tmp_path / "input.jplace"
    path.write_text(json.dumps(doc))
    original = FrechetTreeIndex.random_points
    def forbidden(*args):
        raise AssertionError("Disabled filtering generated a null distribution")
    monkeypatch.setattr(FrechetTreeIndex, "random_points", forbidden)
    baseline = subgenome_consensus(str(path), False, method="frechet", trials=0, random_dist_path="unused.txt")
    assert baseline[1] == doc["placements"]
    monkeypatch.setattr(FrechetTreeIndex, "random_points", original)
    aliased = subgenome_consensus(str(path), False, method="frechet", filter_queries=True, trials=32)
    explicit = subgenome_consensus(str(path), False, method="frechet", post_filter=True, trials=32)
    assert aliased == explicit
    cli = CliRunner().invoke(app, ["subgenome-consensus", "-i", str(path), "-w", "-o", str(tmp_path),
        "--method", "frechet", "--pre-filter", "--post-filter", "--pre-alpha", "0.1", "--post-alpha", "0.5", "--trials", "32"])
    assert cli.exit_code == 0, cli.output + str(cli.exception)
    for stage in ["pre", "post"]:
        with (tmp_path / f"input_{stage}consensus_filter.tsv").open() as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        assert len(rows) == 4
        assert all(r["alpha"] == ("0.1" if stage == "pre" else "0.5") for r in rows)
    saved = json.loads((tmp_path / "input_postconsensus_filtered.jplace").read_text())
    assert len(saved["placements"]) == 3
    assert saved["fields"] == doc["fields"]
    with pytest.raises(ValueError, match="require method"):
        subgenome_consensus(str(path), False, pre_filter=True)


def test_index_reused_partial_empty_groups_mapping_and_weight_invariance(tmp_path, monkeypatch):
    import JEST.frechet as module
    actual = module.FrechetTreeIndex
    constructed = []
    def counted(tree):
        constructed.append(tree)
        return actual(tree)
    monkeypatch.setattr(module, "FrechetTreeIndex", counted)
    doc = document([[(0, 1)], [(0, 1), (10, 1)]], ["good", "bad"])
    mapping = tmp_path / "map.tsv"
    mapping.write_text("good\tA\nbad\tB\n")
    first = filtered(doc, pre_filter=True, post_filter=True, genome_map=str(mapping))
    assert len(constructed) == 1
    assert first[-1]["omitted_groups"] == ["B"]
    assert first[0]["placements"][0]["n"] == ["A"]
    assert len(first[0]["placements"]) == 1
    doc["placements"][1]["p"] = [[9, 1e100, 0], [9, .5e100, 10], [9, .5e100, 10]]
    second = filtered(doc, pre_filter=True, post_filter=True, genome_map=str(mapping))
    assert first[-1] == second[-1]
    assert first[0]["placements"] == second[0]["placements"]
    assert len(constructed) == 2


def test_post_distance_uses_full_distribution_not_only_query_center():
    # Both queries have center 5, but the dispersed query is 5 units from it.
    doc = document([[(5, 1)], [(0, 1), (10, 1)]])
    out, _, _, _, report = filtered(doc, post_filter=True)
    assert out["placements"][0]["p"][0][3] == 5
    assert [s["mean_distance"] for s in report["post_scores"]] == [0, 5]
    assert report["post_filtered_queries"] == ["q1"]
