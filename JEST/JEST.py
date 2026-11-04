'''
JEST.py wraps around the Cpp-binded functions
in Python for better usability
'''
#try:
#    from . import _CTree as CTree
#except ImportError:
#    import _CTree as CTree

from . import _CTree as CTree
from .utils import label_internal_nodes,read_jplace, ERROR_FILTER, ERROR_RANDOM, ERROR_REGEX
import json
import re
import os
import math
from pathlib import Path 

'''
Calculates uncertainty scores for a given jplace file while also providing filtering capabilities
Arguments:
    jplace_path (Required): path to input jplace file
    dest_path (Required): directory path  where results are output (scores, filtered queries and modified jplace files)
    filter_queries (Optional, False by default): whether to filter queries of high uncertainty
    alpha (Optional, 0.05 by default): filter-value cutoff percentage
    random_placements (Optional, 100 by default): number of placements per trial when calculating 
                                                  the uncertainty scores due to random placements
    trials (Optional, 1000 by default): number of random placements trials
    num_threads (Optional, 32  by default): number of parallel threads (for performance purposes)
    tree_path (Optional): unless a path to newick tree is passed, the function uses the tree found in the jplace file
'''
def uncertainty_score(jplace_path: str,
                      write_file: bool,
                      dest_path: str = ".",
                      filter_queries: bool = False,
                      alpha: float = 0.05,
                      random_placements: int = 20,
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
    #print(lwr_idx, edge_num_idx)
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
            #filtered_queries_path = dest_path + "/" + jplace_path.split("/")[-1].split(".")[0]+f"_filtered_queries_{alpha}.txt"
                filtered_queries_path = str(Path(dest_path).joinpath(f"{jplace_path.split("/")[-1].split('.')[0]}_filtered_queries_{alpha}.txt"))
                with open(filtered_queries_path, "w") as f:
                    f.write('\n'.join(filtered_queries))

    return results, remaining_placements

'''
Summarizes multiple placements for a given query to a single placement on the tree
Done on all queries in a jplace file

Arguments:
    jplace_path (Required): path to input jplace file.
    dest_path (Required): file path where results are output.
    gamma (Optional, 1.0 by default): hyperparameter used for consensus calculations.
    num_threads (Optional, 32 by default): number of parallel threads (for performance purposes).
    get_error (Optional, False by default): whether to measure distance of consensus placement to ground truth
                                            (in terms of number of edges); must provide  
                                            - ground truth label  (will throw  an error if there is no label) 
                                            - path to a newick tree (otherwise, the tree found in jplace will be used)
    ground_truth (Optional, Required if get_error is True): the ground truth label to measure the error.
    ground_truth_tree_path (Optional): path to tree to measure the distance between consensus placement and ground truth
                                       will use the jplace tree if node is provided.
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
    '''
    try:
        if (get_error) and (ground_truth == None):
            raise ValueError(ERROR_CONSENSUS_EDGE_ERROR)
    except ValueError as E:
        print(f"Error with Argument Values: {E}")
        exit()
    '''
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
Summarizes multiple placements for all queries in the file to a single placement on the tree
This function assumes that all queries in the file pertain to a single genome.

Arguments:
    jplace_path (Required): path to input jplace file.
    gamma (Optional, 1.0 by default): hyperparameter used for consensus calculations.
    num_threads (Optional, 32 by default): number of parallel threads (for performance purposes).
    get_error (Optional, False by default): whether to measure distance of consensus placement to ground truth
                                            (in terms of number of edges); must provide  
                                            - ground truth label  (will throw  an error if there is no label) 
                                            - path to a newick tree (otherwise, the tree found in jplace will be used)
    ground_truth (Optional, Required if get_error is True): the ground truth label to measure the error.
    ground_truth_tree_path (Optional): path to tree to measure the distance between consensus placement and ground truth
                                       will use the jplace tree if node is provided.
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

    # TODO: add text file feature
    # TODO: add a regex error handling
    try:
        if genome_mapper != "":
            re.compile(genome_mapper)
    except re.error:
        print(ERROR_REGEX)
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
    #placements = jplace_file["placements"]
    dest_file = str(Path(dest_path).joinpath(f"{jplace_path.split("/")[-1].split('.')[0]}_subgenome_error.txt"))
    results = CTree.subgenome_consensus(jplace_tree, tree, placements,num_threads,dest_file, get_error, edge_error, fields, genome_mapper)
    jplace_file["placements"] = results
    if write_file:
        print("Writing to file...")
        new_jplace_path = str(Path(dest_path).joinpath(f"{jplace_path.split("/")[-1].split('.')[0]}_subgenome_consensus.jplace"))
        with open(new_jplace_path, "w") as f:
            json.dump(jplace_file, f, indent=4)
    return results, placements


def placement_error(jplace_path: str,
                         dest_path: str,
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
def placement_edge_error():
    pass
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


def random_placement_uncertainty(tree_path: str,
                      dest_path: str,
                      random_placements: int = 100,
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




