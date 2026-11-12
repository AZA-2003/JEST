'''
JEST.py wraps around the Cpp-binded functions
in Python for better usability while maintaining performance.
'''

from . import _CTree as CTree
from .utils import label_internal_nodes,read_jplace, ERROR_FILTER, ERROR_RANDOM, ERROR_MAPPER
import json
import re
import os
import typer
from typing import Annotated
import math
from pathlib import Path 
import typer

app = typer.Typer(help="JEST library commands",no_args_is_help=True)

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
    - alpha (Optional, 0.05 by default): uncertainty-value cutoff.
    - random_placements (Optional, 10 by default): number of placements per trial when calculating 
                                                  the uncertainty scores due to random placements.
    - trials (Optional, 1000 by default): number of random placements trials.
    - num_threads (Optional, 32  by default): number of parallel threads (for performance purposes)
    - random_path (Optional): unless a path to a null distribution is passed, the function will generate a null 
                            distribution for uncertainty scoring; to produce a file consisting of a null distribution 
                            based on a tree, refer to `random_placement_uncertainty`
Returns the uncertainty scores for each query as well as the remaining queries if filtering was done.
'''
@app.command()
def uncertainty_score(jplace_path: Annotated[str, typer.Option("-i","--in",help = "path to input jplace file.", show_default=False)],
                      write_file:  Annotated[bool, typer.Option("-w","--write-to-file", help = "whether or not an output jplace file should be written.", show_default = False)],
                      dest_path:  Annotated[str, typer.Option("-o","--out", help = "directory path  where results are output.", show_default = True)] = ".",
                      filter_queries: Annotated[bool, typer.Option("-f","--filter", help = "whether to filter queries past an uncertainty threshold.", show_default = True)] = False,
                      alpha: Annotated[float, typer.Option("-a","--alpha", help = "filter-value cutoff percentage.", show_default = True)] = 0.05,
                      random_placements: Annotated[int, typer.Option("-rp","--random-placements", help = "number of placements per one trial of random placement.", show_default = True)] = 10,
                      trials:  Annotated[int, typer.Option("-t","--trials", help = "number of random placements trials.", show_default = True)] = 1000,
                      num_threads:  Annotated[int, typer.Option("-n","--num-threads", help = "number of parallel threads (for performance purposes)", show_default = True)] = 1,
                      random_dist_path: Annotated[str, typer.Option("-r","--random-dist-path", help = "path to a txt file representing the null distribution.", show_default = False)] = ""):
    
    """
    Calculates uncertainty scores for a given jplace file while also providing filtering capabilities.
    """
    if os.path.getsize(jplace_path) == 0:
        print(f"{jplace_path} appears to be empty, exiting now.")
    try:
        if (filter_queries == True) and ((alpha < 0.0) or (alpha > 1.0)):
            raise ValueError(ERROR_FILTER)
        if trials <= 0 or random_placements <= 0:
            raise ValueError(ERROR_RANDOM)
    except ValueError as E:
        print(f"Error with Argument Values: {E}.")
        raise

    #if filter_queries:
    #    trials = max(trials, int(10/alpha))
    try:
        jplace_file = read_jplace(jplace_path)
    except Exception as E:
        raise

    if len(jplace_file["placements"]) == 0:
        print(f"{jplace_path} appears to have no placements, exiting now.")
        return {}

    jplace_tree = jplace_file["tree"]
    jplace_tree = label_internal_nodes(jplace_tree)
    tree = re.sub(r"{[0-9]+}","",jplace_tree)
    is_tree_path = False
    
    # process the mean and std using python instead of numpy to avoid package bloat
    if os.path.isfile(random_dist_path) == False:
        print(f"No random distribution path found! Generating {trials} number of trials for Null distribution.")
        random_placements = list(CTree.get_random_placements_uncertainty(tree,trials,random_placements,num_threads,is_tree_path))
    else:
        print(f"Reading the null distribution from {random_dist_path}.")
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
    print(f"Generating uncertainty scores and writing to {dest_file}.")
    results = CTree.get_uncertainty_pvalue(jplace_tree, tree, 
            jplace_file["placements"], random_placements, 
            num_threads, dest_file,
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
    return results

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
Returns he consensus results as a dictionary
'''
@app.command()
def placement_consensus(jplace_path: Annotated[str, typer.Option("-i", "--in",help = "path to input jplace file.", show_default = False)],
                        write_file: Annotated[bool, typer.Option("-w", "--write-to-file", help = "whether or not an output jplace file should be written.", show_default = False)],
                        dest_path:  Annotated[str, typer.Option("-o","--out", help = "directory path  where results are output.", show_default = True)] = ".",
                        num_threads:  Annotated[int, typer.Option("-n","--num-threads", help = "number of parallel threads (for performance purposes).", show_default = True)] = 1,
                        get_error: Annotated[bool, typer.Option("-e","--get-error", help = "whether to measure weighted distance of consensus to its constituents.", show_default = True)] = False,
                        edge_error: Annotated[bool, typer.Option("-ed", "--edge-error", help = "whether to measure error in terms of number of edges or branch lengths.", show_default = True)] = False):

    """
    Summarizes multiple placements for a given query to a single placement on the tree.
    """
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
    - filter_queries (Optional, False by default): whether to filter queries of high uncertainty.
    - num_threads (Optional, 32 by default): number of parallel threads (for performance purposes).
    - get_error (Optional, False by default): whether to measure distance of consensus placement to each of its constituents.
    - edge_error (Optional, False by default): whether to measure error in terms of number of edges or branch length;
                                            setting it to False means it will use branch length.
    - genome_map (Optional): a regex pattern or file that maps subgenomes to a genome. 
                            This is useful if there are many different genomes represented by the sub-genomes in the file. 
                            Passing nothing assumes that all the subgenomes belong to the same genome. Note that the regex
                            pattern should capture both subgenome and genome groups.
    - multi_copy (Optional, False by default): whether the subgenomes have copies or not; this affects consensus 
                                            calculations as the their contributions are equally split amongst each other.
Returns the consensus results as a dictionary and a list of remaining subgenomes
'''

@app.command()
def subgenome_consensus(jplace_path: Annotated[str, typer.Option("-i", "--in",help = "path to input jplace file.", show_default = False)],
                        write_file: Annotated[bool, typer.Option("-w", "--write-to-file", help = "whether or not an output jplace file should be written.", show_default = False)],
                        dest_path:  Annotated[str, typer.Option("-o","--out", help = "directory path  where results are output.", show_default = True)] = ".",
                      filter_queries: Annotated[bool, typer.Option("-f","--filter", help = "whether to filter queries past an uncertainty threshold.", show_default = True)] = False,
                        alpha: Annotated[float, typer.Option("-a","--alpha", help = "filter-value cutoff percentage.", show_default = True)] = 0.05,
                        random_placements: Annotated[int, typer.Option("-rp","--random-placements", help = "number of placements per one trial of random placement.", show_default = True)] = 10,
                        trials:  Annotated[int, typer.Option("-t","--trials", help = "number of random placements trials.", show_default = True)] = 1000,
                        num_threads:  Annotated[int, typer.Option("-n","--num-threads", help = "number of parallel threads (for performance purposes)", show_default = True)] = 1,
                        random_dist_path: Annotated[str, typer.Option("-r","--random-dist-path", help = "path to a txt file representing the null distribution.", show_default = False)] = "", 
                        get_error: Annotated[bool, typer.Option("-e","--get-error", help = "whether to measure weighted distance of consensus to its constituents.", show_default = True)] = False,
                        edge_error: Annotated[bool, typer.Option("-ed", "--edge-error", help = "whether to measure error in terms of number of edges or branch lengths.", show_default = True)] = False,
                        genome_map: Annotated[str, typer.Option("-g", "--genome-map", help = "regex pattern or file that maps subgenomes to genomes.\n Passing nothing assumes that all subgenomes belong to the same genome.", show_default = True)]= "",
                        multi_copy: Annotated[bool, typer.Option("-m", "--multi-copy", "--multi", help = "whether the subgenomes are single or multicopy.\n Multicopied genomes have their contributions to the consensus split among them.")] = False):

    """
    Summarizes multiple subgenome (contigs, genes, etc..) placements to a single placement on the tree. 
    """
    try:
        if genome_map != "":
            re.compile(genome_map)
    except re.error:
        if os.path.isfile(genome_map) == False:
            print(ERROR_MAPPER)
            exit()
   
    ## TODO add a section to perform random placements to save and reuse that random distribution for later
    jplace_file = read_jplace(jplace_path)
    jplace_tree = jplace_file["tree"]
    jplace_tree = label_internal_nodes(jplace_tree)
    tree = re.sub(r"{[0-9]+}","",jplace_tree)
    
    if os.path.isfile(random_dist_path) == False:
        print(f"No random distribution path found! Generating {trials} number of trials for Null distribution.")
        random_placements = list(CTree.get_random_placements_uncertainty(tree,trials,random_placements,num_threads,False))
    else:
        print(f"Reading the null distribution from {random_dist_path}.")
        with open(random_dist_path, "r") as f:
            random_placements = [float(line.strip()) for line in f]
        if len(random_placements) * alpha < 1:
            raise ValueError(ERROR_ALPHA_RANDOM_MISMATCH)

    mean = sum(random_placements)/len(random_placements)
    std = math.sqrt(sum([(r-mean)*(r-mean) for r in random_placements])/len(random_placements) + 1e-8)
    print(f"Null Distribution generated with mean {mean} and standard deviation {std}") 
    fields = jplace_file["fields"]
    dest_file = str(Path(dest_path).joinpath(f"{jplace_path.split("/")[-1].split('.')[0]}_preconsensus_uncertainty.txt"))
    print(f"Generating uncertainty scores and writing to {dest_file}.")
    results = CTree.get_uncertainty_pvalue(jplace_tree, tree, 
            jplace_file["placements"], random_placements, 
            num_threads, dest_file,
            fields, True)
    print("Done generating uncertainty scores!")
    ## in-place filtration
    print(f"Filtering subgenomes...")
    remaining_placements = []
    filtered_queries = []
    for idx,(name,score) in enumerate(results.items()):
        if score > alpha:
            filtered_queries.append(name)
        else:
            remaining_placements.append(jplace_file["placements"][idx])
    #jplace_file["placements"] = remaining_placements

    print(f"Filtered {len(jplace_file['placements']) - len(remaining_placements)} subgenomes out of {len(jplace_file['placements'])}!") 
    dest_file = str(Path(dest_path).joinpath(f"{jplace_path.split("/")[-1].split('.')[0]}_subgenome_error.txt"))
    results = CTree.subgenome_consensus(jplace_tree, tree, remaining_placements,num_threads,dest_file, get_error, edge_error, fields, genome_map, multi_copy)
     
    jplace_file["placements"] = results["jplace"]
    if write_file:
        print("Writing to file...")
        new_jplace_path = str(Path(dest_path).joinpath(f"{jplace_path.split("/")[-1].split('.')[0]}_subgenome_consensus.jplace"))
        with open(new_jplace_path, "w") as f:
            json.dump(jplace_file, f, indent=4)
    
    if filter_queries: 
        print("Obtaining consensus statistics..")
        dest_file = str(Path(dest_path).joinpath(f"{jplace_path.split("/")[-1].split('.')[0]}_subgenome_stats.txt"))
        subgenome_pvals = CTree.placement_stat_subgenome(jplace_tree, tree, remaining_placements, 
            dict(zip(results["consensus"][0], results["consensus"][1])), 
            num_threads, dest_file, fields, genome_map, 
            random_placements)
        #print(subgenome_pvals)
        print("Filtering subgenomes based on consensus statistics..")
        remaining_placements_post = []
        filtered_queries = []
        for idx,(name,score) in enumerate(subgenome_pvals.items()):
            if score > alpha:
                filtered_queries.append(name)
            else:
                remaining_placements_post.append(next((p for p in remaining_placements if (lambda x:x["n"][0] == name )(p)), None))
        print(f"Filtered {len(remaining_placements) - len(remaining_placements_post)} subgenomes out of {len(remaining_placements)}!") 
    
        jplace_file["placements"] = remaining_placements_post
        if write_file:
            print("Writing to file...")
            new_jplace_path = str(Path(dest_path).joinpath(f"{jplace_path.split("/")[-1].split('.')[0]}_postconsensus_filtered_{alpha}.jplace"))
            with open(new_jplace_path, "w") as f:
                json.dump(jplace_file, f, indent=4)
     
        return results["jplace"], remaining_placements_post
    return results["jplace"], remaining_placements

'''
Calculates placement error for queries in a given jplace file given a single or list of reference placement. 
Placement error consists of a weighted sum of distances between the reference placement and the query's placements.

Arguments:
    - jplace_path (Required): path to input jplace file.
    - dest_file (Required): file pathwhere results are placement errors are output.
    - placement_label (Required) placement label. Can be a single single placement or a file path for multiple placements.
    - tree_path (Optional): unless a path to newick tree is passed, the function uses the tree found in the jplace file.
                           This is useful when the ground truth is not part of the tree used in placement.
    - num_threads (Optional, 1  by default): number of parallel threads (for performance purposes)
'''
@app.command()
def placement_error(jplace_path: Annotated[str, typer.Option("-i", "--in",help = "path to input jplace file.", show_default = False)],
                    dest_file: Annotated[str, typer.Option("-o", "--out",help = "output file path", show_default = False)],
                    placement_label: Annotated[str, typer.Option("-p", "--placement-label", help = "placement label(s); can be a string or a file path")],
                    tree_path: Annotated[str, typer.Option("-t", "--tree-path", help = "path to newick tree. \n Passing nothing means that the function will use the tree in the jplace.",show_default = False)]= "",
                    edge_error: Annotated[bool, typer.Option("-ed", "--edge-error", help = "whether to measure error in terms of number of edges or branch lengths.", show_default = True)]= False,
                   num_threads:  Annotated[int, typer.Option("-n","--num-threads", help = "number of parallel threads (for performance purposes)", show_default = True)] = 1):

    """
    Calculates placement error for queries in a given jplace file given a reference placement. 
    """
    jplace_file = read_jplace(jplace_path)

    jplace_tree = jplace_file["tree"]
    if tree_path == "":
        jplace_tree = label_internal_nodes(jplace_tree)
        tree = re.sub(r"{[0-9]+}","",jplace_tree)
        is_tree_path = False
    else:
        is_tree_path = True
        tree = tree_path

    return CTree.placement_error(jplace_tree, tree, jplace_file["placements"], placement_label, num_threads, edge_error,  dest_path, is_tree_path)

'''
Produces a null distribution of placements given a tree. This is done by simulating trials where a fixed number of nodes areselected and a reference node is chosen amongst them. The average distance between the reference node and the other nodes is then caluclated. The null distribution is the distribution of these scores. This is useful for filtering queries based on uncertainty scores.

Arguments:
    - tree_path (Required): path to a Newick tree or string.
    - dest_file (Required): directory path where results are stored.
    - random_placements (Optional, default is 10): number of nodes selected per trial-- represents the number of placements 
                                                per query.
    - trials (Optional, 1000 by default): number of random placements trials.
    - num_threads (Optional, 1  by default): number of parallel threads (for performance purposes).
'''
@app.command()
def random_placement_uncertainty(tree_path: Annotated[str, typer.Option("-t", "--tree-path", help = "path to newick tree or string.", show_default = False)],                      
                    dest_file: Annotated[str, typer.Option("-o", "--out",help = "output file path", show_default = False)],
                    random_placements: Annotated[int, typer.Option("-rp","--random-placements", help = "number of placements per one trial of random placement.", show_default = True)] = 10,
                    trials:  Annotated[int, typer.Option("-t","--trials", help = "number of random placements trials.", show_default = True)] = 1000,
                    num_threads:  Annotated[int, typer.Option("-n","--num-threads", help = "number of parallel threads (for performance purposes)", show_default = True)] = 1):
    
    """
    Produces a null distribution of placements given a tree.
    """
    ## TODO check if path or string
    if os.path.isfile(tree_path) == True:
        is_tree_path = True
    else:
        is_tree_path = False
    
    # process the mean and std using numpy instead of numpy to avoid package bloat
    random_placements = list(CTree.get_random_placements_uncertainty(tree_path,trials,random_placements,num_threads,is_tree_path))
    mean = sum(random_placements)/len(random_placements)
    std = math.sqrt(sum([(r-mean)*(r-mean) for r in random_placements])/len(random_placements) + 1e-8)
    print(f"Random Placement Distribution generated with mean {mean} and standard deviation {std}") 
    with open(f"{dest_file}","w") as f:
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
    consensus = list(CTree.get_subgenome_consensus(##TODO))    

    ## remove copies far away from consensus
    copy_consensus =  list(CTree.placement_consensus(jplace_tree, tree, placements, num_threads,dest_file, False, False, fields))


placements = uncertainty_score(jplace_path, 
                                write_file = True,
                                filter_queries = True,
                                alpha = alpha,
                                num_threads = num_threads,
                                random_placements = random_placements,
                                trials = trials)[1]
    
    fields = jplace_file["fields"]

'''
if __name__ == "__main__":
        app()
