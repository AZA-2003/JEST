'''
CTree_wrapper.py wraps around the Cpp-binded functions
in Python for better usability
'''
from . import _CTree as CTree
from .utils import label_internal_nodes,read_jplace, ERROR_FILTER, ERROR_RANDOM, ERROR_CONSENSUS_EDGE_ERROR, ERROR_GAMMA
import json
import re


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
                      dest_path: str,
                      filter_queries: bool = False,
                      alpha: float = 0.05,
                      random_placements: int = 100,
                      trials: int = 1000,
                      num_threads: int = 32,
                      tree_path: str = ""):
    
    try:
        if (filter_queries == True) and ((alpha < 0.0) or (alpha > 1.0)):
            raise ValueError(ERROR_FILTER)
        if trials <= 0 or random_placements <= 0:
            raise ValueError(ERROR_RANDOM)
    except ValueError as E:
        print(f"Error with Argument Values: {E}")
        exit()


    jplace_file = read_jplace(jplace_path)

    jplace_tree = jplace_file["tree"]
    if tree_path == "":
        jplace_tree = label_internal_nodes(jplace_tree)
        tree = re.sub(r"{[0-9]+}","",jplace_tree)
        is_tree_path = False
    else:
        is_tree_path = True
        tree = tree_path
    
    # process the mean and std using numpy instead of numpy to avoid package bloat
    random_placements = list(CTree.get_random_placements_uncertainty(tree,trials,random_placements,num_threads,is_tree_path))
    mean = sum(random_placements)/len(random_placements)
    std = (sum([(r-mean)*(r-mean) for r in random_placements])/len(random_placements)) + 1e-8
    print(f"Random Placement Distribution generated with mean {mean} and standard deviation {std}") 
    dest_file = dest_path +"/"+ jplace_path.split("/")[-1].split(".")[0] + "_uncertainty.txt"
    print("Generating uncertainty scores...")
    results = CTree.get_uncertainty_pvalue(jplace_tree,tree,jplace_file["placements"],mean,std,num_threads,dest_file,is_tree_path)
    ## in-place filtration
    print(f"Filtering queries...")
    if filter_queries == True:
        remaining_placements = []
        filtered_queries = []
        for idx,(name,score) in enumerate(results.items()):
            if score > alpha:
                filtered_queries.append(name)
            else:
                remaining_placements.append(jplace_file["placements"][idx])
        jplace_file["placements"] = remaining_placements
        ##TODO Windows support or better string manipulation for writing to destination
        new_jplace_path = dest_path + "/"+ jplace_path.split("/")[-1].split(".")[0]+f"_filtered_{alpha}.jplace"
        with open(new_jplace_path, "w") as f:
            json.dump(jplace_file, f, indent=4)
        if len(filtered_queries) > 0:
            filtered_queries_path = dest_path + "/" + jplace_path.split("/")[-1].split(".")[0]+f"_filtered_queries_{alpha}.txt"
            with open(filtered_queries_path, "w") as f:
                f.write('\n'.join(filtered_queries))

    return results

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
                        dest_path: str,
                        gamma: float = 1.0,
                        num_threads: int = 32,
                        get_error: bool = False,
                        ground_truth: str = None,
                        ground_truth_tree_path: str = ""):
    
    jplace_file = read_jplace(jplace_path)
    jplace_tree = jplace_file["tree"]
    jplace_tree = label_internal_nodes(jplace_tree)
    tree = re.sub(r"{[0-9]+}","",jplace_tree)
    # Only check if there is a ground truth label 
    ## It is possible to use the jplace tree for error estimation
    ## in the the case of inference.
    try:
        if (get_error) and (ground_truth == None):
            raise ValueError(ERROR_CONSENSUS_EDGE_ERROR)
        if (gamma <= 0):
            raise ValueError(ERROR_GAMMA)
    except ValueError as E:
        print(f"Error with Argument Values: {E}")
        exit()

    results =  list(CTree.placement_consensus(jplace_tree, tree, jplace_file["placements"], gamma, num_threads, dest_path, get_error, ground_truth, ground_truth_tree_path))
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

def gene_consensus(jplace_path: str,
                        gamma: float = 1.0,
                        num_threads: int = 32,
                        get_error: bool = False,
                        ground_truth: str = None,
                        ground_truth_tree_path: str = ""):
 
    jplace_file = read_jplace(jplace_path)
    jplace_tree = jplace_file["tree"]
    jplace_tree = label_internal_nodes(jplace_tree)
    tree = re.sub(r"{[0-9]+}","",jplace_tree)
    
    try:
        if (get_error) and (ground_truth == None):
            raise ValueError(ERROR_CONSENSUS_EDGE_ERROR)
        if (gamma <= 0):
            raise ValueError(ERROR_GAMMA)
    except ValueError as E:
        print(f"Error with Argument Values: {E}")
        exit()

    results = CTree.gene_consensus(jplace_tree, tree, jplace_file["placements"], gamma, num_threads, get_error, ground_truth, ground_truth_tree_path)
    return results

