"""Validated equal-query Frechet consensus; all geometry stays in the C++ index.

Each distinct n/nm name is one query. nm abundance is preserved in per-record
output but does not affect the objective. A reusable FrechetTreeIndex is also
available through JEST.CTree for callers managing their own numeric batches.
"""
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import re

from .CTree import FrechetTreeIndex

FIELDS = ["edge_num", "likelihood", "like_weight_ratio", "distal_length", "pendant_length"]


def _real(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _edge_id(value):
    # Accept exact integral JSON numbers, but never round a fractional ID.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("edge_num must be an integer")
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        raise ValueError("edge_num must be an integer")
    value = int(value)
    if not 0 <= value <= 2**64 - 1:
        raise ValueError("edge_num must fit an unsigned 64-bit integer")
    return value


def _names(record):
    if ("n" in record) == ("nm" in record):
        raise ValueError("Every placement record requires exactly one of n or nm")
    values = record.get("n", record.get("nm"))
    if not isinstance(values, list) or not values:
        raise ValueError("Placement names must be a nonempty list")
    if "nm" in record:
        names = []
        for item in values:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("nm entries must be [name, multiplicity]")
            if _real(item[1], "nm multiplicity") <= 0:
                raise ValueError("nm multiplicity must be positive")
            names.append(item[0])
    else:
        names = values
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Query names must be nonempty strings")
    return names


def _membership(names, genome_map):
    if not genome_map:
        return {name: "consensus" for name in names}
    if Path(genome_map).is_file():
        mapping = {}
        with open(genome_map, newline="") as stream:
            for number, row in enumerate(csv.reader(stream, delimiter="\t"), 1):
                if len(row) > 2:
                    raise ValueError("Feature-weighted mapping conflicts with equal-query Frechet weighting")
                if len(row) != 2 or not all(row):
                    raise ValueError(f"Mapping line {number} must contain query<TAB>group")
                if row[0] in mapping and mapping[row[0]] != row[1]:
                    raise ValueError(f"Conflicting mapping for {row[0]!r}")
                mapping[row[0]] = row[1]
        missing = set(names) - mapping.keys()
        if missing:
            raise ValueError(f"Missing query mapping: {sorted(missing)[0]!r}")
        return mapping
    try:
        pattern = re.compile(genome_map)
    except re.error as exc:
        raise ValueError("genome_map must be a file or valid regex") from exc
    if pattern.groups < 2:
        raise ValueError("Grouping regex requires genome (group 1) and subgenome (group 2) captures")
    mapping = {}
    for name in names:
        match = pattern.fullmatch(name)
        if match is None or not match[1] or not match[2]:
            raise ValueError(f"Grouping regex does not capture genome and subgenome for {name!r}")
        mapping[name] = match[1]
    return mapping


def _prepare(document, num_threads):
    """Return the shared index, normalized distributions, and per-record names.

    Validation and normalization occur before parallel solving. The C++ index
    is constructed once and can be reused across numeric batches directly.
    """
    if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads <= 0:
        raise ValueError("num_threads must be a positive integer")
    fields = document.get("fields")
    if not isinstance(fields, list) or any(not isinstance(f, str) for f in fields) or len(set(fields)) != len(fields):
        raise ValueError("fields must be a list of unique field names")
    required = {"edge_num", "like_weight_ratio", "distal_length"}
    if not required <= set(fields):
        raise ValueError("Frechet consensus requires edge_num, like_weight_ratio, and distal_length")
    columns = {field: i for i, field in enumerate(fields)}
    records = document.get("placements")
    if not isinstance(records, list) or not records:
        raise ValueError("No queries supplied for consensus")
    if not isinstance(document.get("tree"), str):
        raise ValueError("jplace tree must be a string")
    index = FrechetTreeIndex(document["tree"])
    distributions, names_by_record, all_names = [], [], set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Placement records must be objects")
        names = _names(record)
        for name in names:
            if name in all_names:
                raise ValueError(f"Duplicate query name: {name!r}")
            all_names.add(name)
        names_by_record.append(names)
        rows = record.get("p")
        if not isinstance(rows, list) or not rows:
            raise ValueError("Each query needs positive total placement weight")
        clean = []
        for row in rows:
            if not isinstance(row, list) or len(row) != len(fields):
                raise ValueError("Placement row size does not match fields")
            edge = _edge_id(row[columns["edge_num"]])
            length = index.edge_length(edge)
            distal = _real(row[columns["distal_length"]], "distal_length")
            tolerance = 32 * math.ulp(1.0) * length
            if distal < -tolerance or distal > length + tolerance:
                raise ValueError(f"distal_length outside edge {edge}")
            distal = max(0.0, min(length, distal))
            weight = _real(row[columns["like_weight_ratio"]], "like_weight_ratio")
            if weight < 0:
                raise ValueError("Placement weights must be nonnegative")
            pendant = 0.0
            if "pendant_length" in columns:
                pendant = _real(row[columns["pendant_length"]], "pendant_length")
                if pendant < 0:
                    raise ValueError("pendant_length must be nonnegative")
            if weight:
                clean.append((edge, distal, weight, pendant))
        if not clean:
            raise ValueError("Each query needs positive total placement weight")
        # Scale first so even finite weights whose sum overflows normalize safely.
        scale = max(row[2] for row in clean)
        total = math.fsum(row[2] / scale for row in clean)
        normalized = [(e, d, (w / scale) / total, p) for e, d, w, p in clean]
        if any(row[2] == 0 for row in normalized):
            raise ValueError("Placement weight dynamic range exceeds double precision")
        distributions.append(normalized)
    return index, distributions, names_by_record


def consensus(document, *, grouped=False, genome_map="", num_threads=1):
    """Compute consensus with one immutable reference index per input batch."""
    prepared = _prepare(document, num_threads)
    return _consensus_prepared(document, prepared, grouped=grouped,
                               genome_map=genome_map, num_threads=num_threads)


def _consensus_prepared(document, prepared, *, grouped, genome_map="", num_threads=1, groups=None):
    index, distributions, names_by_record = prepared
    records = document["placements"]
    columns = {field: i for i, field in enumerate(document["fields"])}
    if groups is None:
        flat_names = [name for names in names_by_record for name in names]
        mapping = _membership(flat_names, genome_map) if grouped else None
        groups = {}
        if grouped:
            for i, names in enumerate(names_by_record):
                for name in names:
                    groups.setdefault(mapping[name], []).append(i)
        else:
            # Identical distributions in one multi-name record are solved once.
            groups = {i: [i] for i in range(len(records))}
    edges, distals, weights, offsets, counts, pendants = [], [], [], [0], [], []
    for members in groups.values():
        counts.append(len(members) if grouped else len(names_by_record[members[0]]))
        pendant_terms = []
        for i in members:
            for edge, distal, weight, pendant in distributions[i]:
                edges.append(edge)
                distals.append(distal)
                weights.append(weight)  # omit common 1/q in the optimizer
                pendant_terms.append((weight / len(members)) * pendant)
        offsets.append(len(edges))
        pendants.append(math.fsum(pendant_terms))
    solved = index.solve_batch(edges, distals, weights, offsets, counts, num_threads)
    output, diagnostics = [], []
    for j, (group, members) in enumerate(groups.items()):
        result = solved[j]
        identity = {"n": [group]} if grouped else {
            key: deepcopy(records[group][key]) for key in ("n", "nm") if key in records[group]
        }
        identity["p"] = [[result.edge_num, 0.0, 1.0, result.distal_length, pendants[j]]]
        output.append(identity)
        diagnostics.append({
            "group": group if grouped else names_by_record[group],
            "consensus_label": index.edge_label(result.edge_num),
            "edge_num": result.edge_num,
            "distal_length": result.distal_length,
            "query_count": result.query_count,
            "positive_row_count": result.positive_row_count if grouped else
                result.positive_row_count * len(names_by_record[group]),
            "mean_squared_distance": result.mean_squared_distance,
            "root_mean_squared_distance": math.sqrt(result.mean_squared_distance),
            "mean_distance": result.mean_distance,
        })
    result_document = deepcopy(document)
    result_document["placements"] = output
    result_document["fields"] = FIELDS.copy()
    metadata = result_document.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("jplace metadata must be an object")
    metadata["frechet"] = {
        "method": "frechet", "weighting": "equal-query; within-query normalized placement weights",
        "names": "each distinct n/nm name is a query; nm abundance ignored for weighting",
        "coordinates": "distal from child; smallest incident original edge ID at contracted vertices",
        "pendant_length": "equal-query mixture mean" if "pendant_length" in columns else "zero placeholder (input unavailable)",
        "likelihood": "zero placeholder; not a sequence likelihood estimate",
        "diagnostics": diagnostics,
    }
    return result_document, diagnostics


def run(path, *, grouped, write_file, dest_path, num_threads, get_error,
        edge_error, genome_map="", multi_copy=False, filter_queries=False,
        return_diagnostics=False, pre_filter=False, post_filter=False,
        pre_alpha=None, post_alpha=None, alpha=0.05, trials=1000,
        random_placements=10, random_seed=48, random_dist_path=""):

    if multi_copy:
        raise ValueError("multi_copy conflicts with equal-query Frechet weighting")
    post_filter = post_filter or filter_queries
    if (pre_filter or post_filter) and not grouped:
        raise ValueError("Pre/post consensus filtering requires grouped consensus")
    if edge_error:
        raise ValueError("Frechet diagnostics use continuous branch-length distance, not edge-count error")
    with open(path) as stream:
        document = json.load(stream)
    filtering = None
    remaining = before_consensus = document["placements"]
    if pre_filter or post_filter:
        from .frechet_filtering import filtered_consensus
        result, diagnostics, remaining, before_consensus, filtering = filtered_consensus(
            document, genome_map=genome_map, num_threads=num_threads,
            pre_filter=pre_filter, post_filter=post_filter,
            pre_alpha=alpha if pre_alpha is None else pre_alpha,
            post_alpha=alpha if post_alpha is None else post_alpha,
            trials=trials, random_placements=random_placements,
            random_seed=random_seed, random_dist_path=random_dist_path)
    else:
        result, diagnostics = consensus(document, grouped=grouped, genome_map=genome_map, num_threads=num_threads)
    prefix = Path(path).stem
    kind = "subgenome" if grouped else "placement"
    if write_file or get_error:
        Path(dest_path).mkdir(parents=True, exist_ok=True)
        # A label describes the selected edge's child; the coordinate is needed
        # to identify the continuous consensus, including at a proximal endpoint.
        name_column = "group" if grouped else "query"
        with open(Path(dest_path) / f"{prefix}_{kind}_consensus.tsv", "w", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow([name_column, "consensus_label", "edge_num", "distal_length"])
            for diagnostic in diagnostics:
                names = [diagnostic["group"]] if grouped else diagnostic["group"]
                for name in names:
                    writer.writerow([name, diagnostic["consensus_label"],
                                     diagnostic["edge_num"], diagnostic["distal_length"]])
    if write_file:
        with open(Path(dest_path) / f"{prefix}_{kind}_consensus.jplace", "w") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
    if get_error:
        with open(Path(dest_path) / f"{prefix}_{kind}_frechet.tsv", "w", newline="") as stream:
            diagnostic_fields = ["group", "consensus_label", "edge_num", "distal_length", "query_count",
                                 "positive_row_count", "mean_squared_distance", "root_mean_squared_distance", "mean_distance"]
            writer = csv.DictWriter(stream, fieldnames=diagnostic_fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(diagnostics)
    if filtering is not None:
        from .frechet_filtering import write_filter_outputs
        write_filter_outputs(document, filtering, before_consensus, remaining,
                             dest_path=dest_path, prefix=prefix,
                             write_file=write_file, get_error=get_error)
    if return_diagnostics:
        out = {"placements": result["placements"], "diagnostics": diagnostics}
        if grouped:
            out["remaining_placements"] = remaining
        if filtering is not None:
            out["filtering"] = filtering
            out["preconsensus_placements"] = before_consensus
        return out
    if grouped:
        return result["placements"], remaining
    return result["placements"]
