"""Point-aware Fréchet filtering with separate, reproducible null calibrations.

Scores are mean branch-length distances, never MSE disguised as uncertainty.
A null query has random_placements equally weighted attachments sampled
uniformly along total reference branch length. Null percentiles measure position
in that reference distribution; they are not upper-tail significance p-values.
"""
from bisect import bisect_left
from copy import deepcopy
import csv
import json
from pathlib import Path

from .frechet import _prepare, _consensus_prepared, _membership, _real

SCORE_FIELDS = ["query", "group", "mean_distance", "null_percentile", "alpha", "filtered",
                "reference_edge_num", "reference_distal_length"]


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _cutoff(value, name):
    value = _real(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _buffers(distributions):
    edges, distal, weights, offsets = [], [], [], [0]
    for distribution in distributions:
        for edge, distance, weight, _ in distribution:
            edges.append(edge)
            distal.append(distance)
            weights.append(weight)
        offsets.append(len(edges))
    return edges, distal, weights, offsets


def _select_records(records, retained):
    selected = []
    for record in records:
        key = "n" if "n" in record else "nm"
        names = [item for item in record[key] if (item if key == "n" else item[0]) in retained]
        if names:
            copy = deepcopy(record)
            copy[key] = names
            selected.append(copy)
    return selected


def _score(name, group, raw, null, alpha, reference):
    # Strict lower-tail rank makes ties conservative, including an all-zero tree.
    percentile = bisect_left(null, raw) / len(null)
    return {"query": name, "group": group, "mean_distance": raw,
            "null_percentile": percentile, "alpha": alpha, "filtered": percentile > alpha,
            "reference_edge_num": reference["edge_num"],
            "reference_distal_length": reference["distal_length"]}


def filtered_consensus(document, *, genome_map, num_threads, pre_filter, post_filter,
                       pre_alpha, post_alpha, trials, random_placements, random_seed,
                       random_dist_path=""):
    if random_dist_path:
        raise ValueError("Legacy random_dist_path scores are not calibrated for Frechet filtering; omit it to generate stage-specific null distributions")
    _positive_integer(trials, "trials")
    _positive_integer(random_placements, "random_placements")
    if pre_filter and random_placements < 2:
        raise ValueError("Pre-consensus filtering requires at least two random_placements")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or not 0 <= random_seed < 2**64:
        raise ValueError("random_seed must be an unsigned 64-bit integer")
    if pre_filter:
        pre_alpha = _cutoff(pre_alpha, "pre_alpha")
    if post_filter:
        post_alpha = _cutoff(post_alpha, "post_alpha")
    prepared = _prepare(document, num_threads)
    index, distributions, names_by_record = prepared
    names = [name for row in names_by_record for name in row]
    mapping = _membership(names, genome_map)
    original_groups = list(dict.fromkeys(mapping[name] for name in names))
    # Sample once outside workers, then reuse these points for both calibrations.
    samples = index.random_points(trials * random_placements, random_seed)
    null_edges = [p.edge_num for p in samples]
    null_distal = [p.distal_length for p in samples]
    null_weights = [1 / random_placements] * len(null_edges)
    null_offsets = list(range(0, len(null_edges) + 1, random_placements))
    null_buffers = (null_edges, null_distal, null_weights, null_offsets)
    pre_scores, post_scores = [], []
    retained = set(names)
    if pre_filter:
        _, query_diagnostics = _consensus_prepared(document, prepared, grouped=False, num_threads=num_threads)
        null_results = index.solve_batch(*null_buffers, [1] * trials, num_threads)
        pre_null = sorted(r.mean_distance for r in null_results)
        for i, row in enumerate(names_by_record):
            reference = query_diagnostics[i]
            for name in row:
                score = _score(name, mapping[name], reference["mean_distance"], pre_null, pre_alpha, reference)
                pre_scores.append(score)
                if score["filtered"]:
                    retained.remove(name)
    before_consensus = _select_records(document["placements"], retained)
    groups = {}
    for i, row in enumerate(names_by_record):
        for name in row:
            if name in retained:
                groups.setdefault(mapping[name], []).append(i)
    result, diagnostics = _consensus_prepared(document, prepared, grouped=True,
                                              num_threads=num_threads, groups=groups)
    if post_filter and groups:
        references = {d["group"]: d for d in diagnostics}
        entries = [(name, i) for i, row in enumerate(names_by_record) for name in row if name in retained]
        buffers = _buffers(distributions[i] for _, i in entries)
        ref_edges = [references[mapping[name]]["edge_num"] for name, _ in entries]
        ref_distal = [references[mapping[name]]["distal_length"] for name, _ in entries]
        raw_scores = index.mean_distance_batch(*buffers, ref_edges, ref_distal, num_threads)
        # Only one null score array is resident per group, not groups * trials.
        by_group = {}
        for (name, _), raw in zip(entries, raw_scores):
            by_group.setdefault(mapping[name], []).append((name, raw))
        scores_by_name = {}
        for group, entries_in_group in by_group.items():
            reference = references[group]
            null = sorted(index.mean_distance_batch(
                *null_buffers, [reference["edge_num"]] * trials,
                [reference["distal_length"]] * trials, num_threads))
            for name, raw in entries_in_group:
                score = _score(name, group, raw, null, post_alpha, reference)
                scores_by_name[name] = score
                if score["filtered"]:
                    retained.remove(name)
        post_scores = [scores_by_name[name] for name, _ in entries]
    remaining = _select_records(document["placements"], retained)
    filtering = {
        "pre_filter": pre_filter, "post_filter": post_filter,
        "pre_alpha": pre_alpha if pre_filter else None,
        "post_alpha": post_alpha if post_filter else None,
        "statistic": "normalized placement-weighted mean branch-length distance",
        "pre_reference": "each query's own Frechet consensus",
        "post_reference": "group Frechet consensus after pre-filtering; not refitted after post-filtering",
        "null_sampling": "independent uniform branch-length attachments with equal weights; includes positive root stem",
        "percentile": "fraction of null scores strictly less than observed; filter when percentile > alpha",
        "random_seed": random_seed, "trials": trials, "random_placements": random_placements,
        "pre_scores": pre_scores, "post_scores": post_scores,
        "pre_filtered_queries": [s["query"] for s in pre_scores if s["filtered"]],
        "post_filtered_queries": [s["query"] for s in post_scores if s["filtered"]],
        "omitted_groups": [g for g in original_groups if g not in groups],
    }
    result["metadata"]["frechet"]["filtering"] = filtering
    return result, diagnostics, remaining, before_consensus, filtering


def write_filter_outputs(document, filtering, before_consensus, remaining, *,
                         dest_path, prefix, write_file, get_error):
    if not (write_file or get_error):
        return
    for stage, records in [("pre", before_consensus), ("post", remaining)]:
        if not filtering[f"{stage}_filter"]:
            continue
        with open(Path(dest_path) / f"{prefix}_{stage}consensus_filter.tsv", "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SCORE_FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows(filtering[f"{stage}_scores"])
        if write_file:
            filtered = deepcopy(document)
            filtered["placements"] = records
            filtered.setdefault("metadata", {})["frechet_filtering"] = filtering
            with open(Path(dest_path) / f"{prefix}_{stage}consensus_filtered.jplace", "w") as stream:
                json.dump(filtered, stream, indent=2, allow_nan=False)
