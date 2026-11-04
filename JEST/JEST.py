'''
JEST.py wraps around the Cpp-binded functions
in Python for better usability while maintaining performance.
'''

from . import _CTree as CTree
from .utils import label_internal_nodes,read_jplace, ERROR_FILTER, ERROR_RANDOM, ERROR_REGEX
import json
import re
import os
import math
from pathlib import Path 

'''
Calculates uncertainty scores for a given jplace file while also providing filtering capabilities based on a hyperparameter.
Returns a modified jplace file with uncertain queries filtered out, a text file of the uncertainty scores of the queries and 
a list of queary names that were filtered out (if any).

Arguments:
    - jplace_path (Required): path to input jplace file.
    - write_file (Required): whether or not an output jplace file should be written.
    - dest_path (Optional, current path by default): directory path  where results are output (scores, 
                            filtered queries and modified jplace files).
    - filter_queries (Optional, False by default): whether to filter queries of high uncertainty.
    - alpha (Optional, 0.05 by default): filter-value cutoff percentage.
    - random_placements (Optional, 10 by default): number of placements per trial when calculating 
                                                  the uncertainty scores due to random placements.
    - trials (Optional, 1000 by default): number of random placements trials.
    - num_threads (Optional, 32  by default): number of parallel threads (for performance purposes)
    - tree_path (Optional): unless a path to newick tree is passed, the function uses the tree found in the jplace file.
    - random_path (Optional): unless a path to a null distribution is passed, the function will generate a null 
                            distribution for uncertainty scoring; to produce a file consisting of a null distribution 
                            based on a tree, refer to `random_placement_uncertainty`
'''
def uncertainty_score(jplace_path: str,
                      write_file: bool,
                      dest_path: str = ".",
                      filter_queries: bool = False,
                      alpha: float = 0.05,
                      random_placements: int = 10,
                      trials: int = 1000,
                      num_threads: int = 32,
                      tree_path: str = "",
                      random_dist_path = ""):
    
    if os.path.getsize(jplace_path) == 0:
        print(f"{jplace_path} appears to be empty, exiting now.")
    try:
        if (filter_queries == True) and ((alpha < 0.0) or (alpha > 1.0)):
            raise ValueError(ERROR_FILTER)
        if trials <= 0 or random_placements <= 0:
            raise ValueError(ERROR_RANDOM)
    except ValueError as E:
        print(f"Error with Argument Values: {E}.")
        exit()

    if filter_queries:
        trials = max(trials, 10/alpha)
    
    jplace_file = read_jplace(jplace_path)
    if len(jplace_file["placements"]) == 0:
        print(f"{jplace_path} appears to have no placements, exiting now.")

    jplace_tree = jplace_file["tree"]
    if tree_path == "":
        jplace_tree = label_internal_nodes(jplace_tree)
        tree = re.sub(r"{[0-9]+}","",jplace_tree)
        is_tree_path = False
    else:
        is_tree_path = True
        tree = tree_path
    
    # process the mean and std using numpy instead of numpy to avoid package bloat
    if random_dist_path == "":
        print(f"Generating {trials} number of trials for Null distribution.")
        random_placements = list(CTree.get_random_placements_uncertainty(tree,trials,random_placements,num_threads,is_tree_path))
    else:
        print(f"Reading the Null distribution from {random_dist_path}.")
        with open(random_dist_path, "r") as f:
            random_placements = [float(line.strip()) for line in f]
        if len(random_placements) * alpha < 1:
            raise ValueError(ERROR_ALPHA_RANDOM_MISMATCH)

    mean = sum(random_placements)/len(random_placements)
    std = math.sqrt(sum([(r-mean)*(r-mean) for r in random_placements])/len(random_placements) + 1e-8)
    print(f"Null Distribution generated with mean {mean} and standard deviation {std}") 
    #dest_file = dest_path +"/"+ jplace_path.split("/")[-1].split(".")[0] + "_uncertainty.txt"
    fields = jplace_file["fields"]
    lwr_idx = fields.index("like_weight_ratio")
    edge_num_idx = fields.index("edge_num")
    dest_file = str(Path(dest_path).joinpath(f"{jplace_path.split("/")[-1].split('.')[0]}_uncertainty.txt"))
    #print(f"Generating uncertainty scores and writing to {dest_file}.")
    results = CTree.get_uncertainty_pvalue(jplace_tree, tree, 
            jplace_file["placements"], random_placements, 
            num_threads, dest_file, is_tree_path,
            fields, False)
    print("Done generating uncertainty scores!")
    ## in-place filtration
    if filter_queries == True:
        print(f"Filtering queries...")
        remaining_placements = []
        filtered_queries = []
        for idx,(name,score) in enumerate(results.items()):
            if score > alpha:
                filtered_queries.append(name)
            else:
                remaining_placements.append(jplace_file["placements"][idx])
        jplace_file["placements"] = remaining_placements
        if write_file:
            print("Writing to file...")
            new_jplace_path = str(Path(dest_path).joinpath(f"{jplace_path.split("/")[-1].split('.')[0]}_filtered_{alpha}.jplace"))
            with open(new_jplace_path, "w") as f:
                json.dump(jplace_file, f, indent=4)
            if len(filtered_queries) > 0:
                filtered_queries_path = str(Path(dest_path).joinpath(f"{jplace_path.split("/")[-1].split('.')[0]}_filtered_queries_{alpha}.txt"))
                with open(filtered_queries_path, "w") as f:
                    f.write('\n'.join(filtered_queries))

    return results, remaining_placements

'''
Summarizes multiple placements for a given query to a single placement on the tree.
Returns a modified jplace of the consensus placements as well as a text file with the weighted sum of distances 
between the consensus and the query's placements.

Arguments:
    - jplace_path (Required): path to input jplace file.
    - write_file (Required): whether or not an output jplace file should be written.
    - dest_path (Optional, current path by default): directory path  where results are output (errors, modified jplace files).
    - num_threads (Optional, 32 by default): number of parallel threads (for performance purposes).
    - get_error (Optional, False by default): whether to measure weighted distance of consensus placement to its constituents.
                                       will use the jplace tree if node is provided.
    - edge_error (Optional, False by default): whether to measure error in terms of number of edges or branch length;
                                            setting it to False means it will use branch length.
'''

def placement_consensus(jplace_path: str,
                        write_file: bool,
                        dest_path: str = ".",
                        num_threads: int = 32,
                        get_error: bool = False,
                        edge_error: bool = False):

    jplace_file = read_jplace(jplace_path)
    jplace_tree = jplace_file["tree"]
    jplace_tree = label_internal_nodes(jplace_tree)
    tree = re.sub(r"{[0-9]+}","",jplace_tree)
    fields = jplace_file["fields"]
    placements = jplace_file["placements"]
    dest_file = str(Path(dest_path).joinpath(f"{jplace_path.split("/")[-1].split('.')[0]}_placement_error.txt"))
    results =  CTree.placement_consensus(jplace_tree, tree, placements, num_threads,dest_file, get_error, edge_error, fields)
    jplace_file["placements"] = results 
    if write_file:
        print("Writing to file...")
        new_jplace_path = str(Path(dest_path).joinpath(f"{jplace_path.split("/")[-1].split('.')[0]}_placement_consensus.jplace"))
        with open(new_jplace_path, "w") as f:
            json.dump(jplace_file, f, indent=4)
    return results

'''
Summarizes multiple subgenome placements to a single placement on the tree. Multiple genomes with their own subgenomes can 
exist in the file provided a text file mapping the subgenomes to their belonging genomes or a regex pattern. Returns a modified 
jplace of the consensus placements as well as a text file with the weighted sum of distances between the consensus and the 
subgenomes.

Arguments:
    - jplace_path (Required): path to input jplace file.
    - write_file (Required): whether or not an output jplace file should be written.
    - dest_path (Optional, current path by default): directory path  where results are output (errors, modified jplace files).
    - num_threads (Optional, 32 by default): number of parallel threads (for performance purposes).
    - get_error (Optional, False by default): whether to measure weighted distance of consensus placement to its constituents.
                                       will use the jplace tree if node is provided.
    - edge_error (Optional, False by default): whether to measure error in terms of number of edges or branch length;
                                            setting it to False means it will use branch length.
    - genome_map (Optional): a regex pattern or file that maps subgenomes to a genome. This is useful if there are many different                           genomes represented by the sub-genomes in the file. Passing nothing assumes that all the subgenomes be                           long to the same genome.
'''

def subgenome_consensus(jplace_path: str,
                   write_file: bool,
                   dest_path: str = "",
                   alpha: float = 0.05,
                   random_placements: int = 20,
                   trials: int = 1000,
                   num_threads: int = 32,
                   get_error: bool = False,
                   edge_error: bool = False,
                   genome_mapper: str = ""):

    try:
        if genome_mapper != "":
            re.compile(genome_mapper)
    except re.error:
        if os.path.isfile(genome_mapper) == False:
            print(ERROR_MAPPER)
            exit()

    placements = uncertainty_score(jplace_path, 
                                write_file = False,
                                filter_queries = True,
                                alpha = alpha,
                                num_threads = num_threads,
                                random_placements = random_placements,
                                trials = trials)[1]
    
    jplace_file = read_jplace(jplace_path)
    jplace_tree = jplace_file["tree"]
    jplace_tree = label_internal_nodes(jplace_tree)
    tree = re.sub(r"{[0-9]+}","",jplace_tree)
    
    fields = jplace_file["fields"]
    dest_file = str(Path(dest_path).joinpath(f"{jplace_path.split("/")[-1].split('.')[0]}_subgenome_error.txt"))
    results = CTree.subgenome_consensus(jplace_tree, tree, placements,num_threads,dest_file, get_error, edge_error, fields, genome_mapper)
    jplace_file["placements"] = results
    if write_file:
        print("Writing to file...")
        new_jplace_path = str(Path(dest_path).joinpath(f"{jplace_path.split("/")[-1].split('.')[0]}_subgenome_consensus.jplace"))
        with open(new_jplace_path, "w") as f:
            json.dump(jplace_file, f, indent=4)
    return results, placements


'''
Calculates placement error for queries in a given jplace file given a single or list of ground truth. 
Placement error consists of a weighted sum of distances between the ground truth genome and the query's placements.

Arguments:
    - jplace_path (Required): path to input jplace file.
    - dest_file (Required): file pathwhere results are placement errors are output.
    - ground_truth (Required) ground truth label. Can be a single string or a list of strings.
    - tree_path (Optional): unless a path to newick tree is passed, the function uses the tree found in the jplace file.
                           This is useful when the ground truth is not part of the tree used in placement.
    - num_threads (Optional, 32  by default): number of parallel threads (for performance purposes)
'''

def placement_error(jplace_path: str,
                    dest_file: str,
                    ground_truth: str | list[str],
                    tree_path: str = "",
                    edge_error: bool = False,
                    num_threads: int = 32):

    jplace_file = read_jplace(jplace_path)

    jplace_tree = jplace_file["tree"]
    if tree_path == "":
        jplace_tree = label_internal_nodes(jplace_tree)
        tree = re.sub(r"{[0-9]+}","",jplace_tree)
        is_tree_path = False
    else:
        is_tree_path = True
        tree = tree_path

    if type(ground_truth) is list: 
        return CTree.placement_error_multilabel(jplace_tree, tree, jplace_file["placements"], ground_truth, num_threads, edge_error,  dest_path, is_tree_path)
    else:
        return CTree.placement_error(jplace_tree, tree, jplace_file["placements"], ground_truth, num_threads, edge_error,  dest_path, is_tree_path)

'''
Produces a null distribution of placements given a tree. This is done by simulating trials where a fixed number of nodes areselected and a reference node is chosen amongst them. The average distance between the reference node and the other nodes is then caluclated. The null distribution is the distribution of these scores. This is useful for filtering queries based on uncertainty scores.

Arguments:
    - tree_path (Required): path to a Newick tree.
    - dest_path (Required): directory path where results are stored.
    - random_placements (Optional, default is 10): number of nodes selected per trial-- represents the number of placements 
                                                per query.
    - trials (Optional, 1000 by default): number of random placements trials.
    - num_threads (Optional, 32  by default): number of parallel threads (for performance purposes).
'''

def random_placement_uncertainty(tree_path: str,
                      dest_path: str,
                      random_placements: int = 10,
                      trials: int = 1000,
                      num_threads: int = 32):
    
    is_tree_path = True
    tree = tree_path
    
    # process the mean and std using numpy instead of numpy to avoid package bloat
    random_placements = list(CTree.get_random_placements_uncertainty(tree,trials,random_placements,num_threads,is_tree_path))
    mean = sum(random_placements)/len(random_placements)
    std = math.sqrt(sum([(r-mean)*(r-mean) for r in random_placements])/len(random_placements) + 1e-8)
    print(f"Random Placement Distribution generated with mean {mean} and standard deviation {std}") 
    with open(f"{dest_path}","w") as f:
        for rp in sorted(random_placements):
            f.write(f"{rp}\n")
    return random_placements 

'''
def multi_copy_consensus(jplace_path: str,
                         write_file: bool,
                         dest_path: str= ".",
                         alpha: float = 0.05,
                         random_placements: int = 100,
                         trials: int = 1000,
                         num_threads: int = 32, 
                         genome_mapper: str = ""):
    
    jplace_file = read_jplace(jplace_path)
    jplace_tree = jplace_file["tree"]
    fields = jplace_file["fields"]
    jplace_tree = label_internal_nodes(jplace_tree)
    tree = re.sub(r"{[0-9]+}","",jplace_tree)
    
    try:
        if ((alpha < 0.0) or (alpha > 1.0)):
            raise ValueError(ERROR_FILTER)
        if trials <= 0 or random_placements <= 0:
            raise ValueError(ERROR_RANDOM)
    except ValueError as E:
        print(f"Error with Argument Values: {E}")
        exit()
    consensus, remaining_placements = subgenome_consensus(jplace_path,
                                        write_file = False,
                                        alpha = alpha,
                                        random_placements = random_placements,
                                        trials = trials,
                                        num_threads = num_threads,
                                        get_error = True,
                                        edge_error = False,
                                        genome_mapper = genome_mapper)

    ## remove copies far away from consensus
    copy_consensus =  list(CTree.placement_consensus(jplace_tree, tree, placements, num_threads,dest_file, False, False, fields))


'''
