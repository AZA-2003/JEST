from . import _CTree as CTree
from multiprocessing import Pool
import json
import os
import re
import time



def uncertainty_score_file(jplace_path: str,
                           dest_dir: str,
                           tree_path: str,
                           filter_queries: bool,
                           rp_mean: float,
                           rp_std: float,
                           alpha: float,
                           random_placements: int,
                           trials: int,
                           num_threads: int):
    
    with open(jplace_path,"r") as f:
        jplace_file = json.load(f)
    
    jplace_tree = jplace_file["tree"]
    print(f"#####{jplace_path.split(".jplace")[0]}#####")
    dest_path = dest_dir + jplace_path.split(".jplace")[0] + f"_uncertainty_scores.txt"
    results = CTree.get_uncertainty_pvalue(jplace_tree,tree_path,jplace_file["placements"], rp_mean, rp_std, num_threads, dest_path, True)
    if filter_queries:
        remaining_placements = []
        filtered_queries = []
        for idx,(name,score) in enumerate(results):
            if score <= 1-alpha:
                filtered_queries.append(name)
            else:
                remaining_placements.append(jplace_file["placements"][idx])
        jplace_file["placements"] = remaining_placements
        new_jplace_path = jplace_path.split(".jplace")[0]+f"_filtered_{alpha}.jplace"
        with open(new_jplace_path, "w") as f:
            json.dump(jplace_file, f, indent=4)

        filtered_queries_path = jplace_path.split(".jplace")[0]+f"_filtered_queries_{alpha}.txt"
        with open(filtered_queries_path, "w") as f:
            f.write('\n'.join(filtered_reads))

