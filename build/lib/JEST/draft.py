import treeswift as ts
import json
import re
import argparse
import numpy as np
from scipy import stats
from .utils import read_jplace, label_internal_nodes, ERROR_FILTER, ERROR_RANDOM
## temp imports to get some metrics
import time
'''
Generates a number of trials where random placements are made
'''
def random_placement_score(tree, trials, random_placements,
                            alpha, verbose):
    lbl_to_node = tree.label_to_node("all")
    uncertainties = []
    for __ in range(trials):
        tree_nodes = np.array(list(tree.labels()))
        placements = np.random.choice(tree_nodes,size=random_placements,replace=True)
        ref_placement = np.random.choice(placements,1)[0]
        ref_placement_idx = list(placements).index(ref_placement)
        ref_placement = lbl_to_node[ref_placement]
        node_placements = [lbl_to_node[p] for p in placements]
        ref_distance = [np.abs(tree.distance_between(p,ref_placement)) for p in node_placements]
        uncertainties.append(sum(ref_distance)/random_placements)
    mean,std = np.mean(uncertainties), np.std(uncertainties)
    norm_uncertainties = (np.array(uncertainties) - mean) / (std + 1e-8)
    if alpha != -1:
        filter_threshold = np.percentile(norm_uncertainties,alpha*100)
        if verbose:
            print(f"filter threshold value is {filter_threshold}")
    else:
        filter_threshold = np.inf
    ## Added a verbose feature to print out stats
    if verbose:
        print("### Normalized stats ###")
        print(f"1 percentile is {np.percentile(norm_uncertainties,1)}")
        #print(f"0.1 percentile is {np.percentile(norm_uncertainties,0.1)}")
        print(f"25 percentile is {np.percentile(norm_uncertainties,25)}")
        print(f"50 percentile is {np.percentile(norm_uncertainties,50)}")
        print(f"75 percentile is {np.percentile(norm_uncertainties,75)}")
        print(f"90 percentile is {np.percentile(norm_uncertainties,90)}")
        print("### Normalizing factors ###")
        print(f"mean of random distribution is {mean}")
        print(f"standard deviation of random distribution is {std}")
    #print(f"0.5 lies on {stats.percentileofscore(norm_uncertainties,0.5)} percentile")
    
    #norm_uncertainties = np.array(uncertainties)
    #print(norm_uncertainties)
    #norm_uncertainties /= np.median(uncertainties)
    #norm_uncertainties /= np.mean(uncertainties)
    
    return norm_uncertainties, mean, std, filter_threshold

## CREATE A MULTITHREADED VERSION OF uncertainty_score

'''
takes in a path for jplace and an output destination file to write the uncertainty score of all 
query in the jplace file
'''
def uncertainty_score(jplace_path : str, dest_path : str, alpha : float = -1.0,
        normalize: bool = True, trials : int = 10000, 
        random_placements : int = 100, rooted_tree : bool = False,
        verbose : bool = False):

    try:
        if (alpha < 0.0 and alpha != -1.0) or (alpha > 1.0):
            raise ValueError(ERROR_FILTER)
        if trials <= 0 or random_placements <= 0:
            raise ValueError(ERROR_RANDOM)
    except ValueError as E:
        print(f"Error with Argument Values: {E}")
        exit()

    jd = read_jplace(jplace_path)
    jtree = jd["tree"]
    #jtree = tree.newick()
    try:
        jtree = label_internal_nodes(jtree)
        jtree_r = re.sub(r"{[0-9]+}","",jtree)
        tree = ts.read_tree_newick(jtree_r)            
    except Exception as E:
        print("Error reading the tree!")
        exit()
    
    mean = 0.0
    std = 1.0
    filtered_reads = []
    filter_threshold = np.inf
    ###
    begin = time.time()
    ###
    if normalize:
        random_pl, mean, std, filter_threshold = random_placement_score(tree, trials, random_placements, 
                                                                    alpha, verbose)
    ###
    print(f"Time taken for random placements {time.time() - begin} seconds")
    ###
    try:
        fields = jd["fields"]
        edge_num = fields.index("edge_num")
        l_ratio = fields.index("like_weight_ratio")
    except Exception as E:
        print(NO_FIELDS)
        edge_num = 0
        l_ratio = 2
    lbl_to_node = tree.label_to_node("all")
    ###
    begin = time.time()
    num_placements = len(jd["placements"])
    ###
    with open(dest_path,"w") as f:
        f.write(f"name\tuncertainty\tuncertainty_ratio\tpercentile\tp-value\n")
        ## iterating over all queries
        #for placement in jd["placements"]:
        idx = 0
        while idx != len(jd["placements"]):
            placement = jd["placements"][idx]
            name = placement['n'][0]
            p_l_ratio = []
            p_edge_num =[]
            p_lbl = []
            ## gather the likeihood and edge number
            if len(placement['p']) == 0:
                f.write(
                        f"{name}\tNaN\tNaN\tNaN\n"
                        )
                idx += 1
                continue
            for p_idx in range(len(placement['p'])):
            #for p_idx in range(min(len(placement['p']),20)):
                edge = placement['p'][p_idx][edge_num]
                ixe = jtree.find("{" + f"{edge}" + "}")
                ixs = max( 
                            (
                                (jtree[:ixe].rfind(",")),
                                (jtree[:ixe].rfind("(")),
                                (jtree[:ixe].rfind(")")),    
                            )
                        )
                likelihood = float(placement['p'][p_idx][l_ratio])
                lbl_placement, blen = jtree[ixs+1:ixe].split(":")
                #lbl_placement  = jtree[ixs+1:ixe].split(":")[0]
                #print(lbl_placement,blen)
                p_edge_num.append(edge)
                p_l_ratio.append(likelihood)
                p_lbl.append(lbl_placement)
           
            #print(p_lbl)
            p_l_ratio = np.array(p_l_ratio)
            p_l_ratio /= np.sum(p_l_ratio)
            
            node_placements = [lbl_to_node[p] for i,p in enumerate(p_lbl)]
            if rooted_tree:
                ref_placement = tree.mrca(p_lbl)
                ref_distance = [np.abs(tree.distance_between(p,ref_placement)) for p in node_placements]
                weighted_score = sum([p_l_ratio[i]*ref_distance[i] for i in range(len(p_l_ratio))])
            else:
                ref_placement_idx = np.argmax(p_l_ratio)
                ref_placement = p_lbl[ref_placement_idx]
                ref_placement = lbl_to_node[ref_placement]
                ref_distance = [np.abs(tree.distance_between(p,ref_placement)) for p in node_placements]
                weighted_score = sum([p_l_ratio[i]*ref_distance[i] for i in range(len(p_l_ratio))])/(1-p_l_ratio[ref_placement_idx]) if p_l_ratio[ref_placement_idx] != 1.0 else 0.0
            
            weighted_score_norm = (weighted_score - mean)/(std+1e-8)
            proportion = stats.percentileofscore(random_pl,weighted_score_norm)/100
            pvalue = round(1-stats.norm.cdf(weighted_score_norm),4)
            f.write(
                    f"{name}\t{weighted_score}\t{weighted_score_norm}\t{proportion}\t{pvalue}\n"
                    )
            #if proportion > filter_threshold:
            if pvalue <= 1-alpha:
                filtered_reads.append(name)
                jd["placements"].pop(idx)
            else:
                idx += 1
    ###
    end = time.time()
    print(f"Time taken to produce unncertainty scores ({num_placements} placements): {end-begin} seconds")
    print(f"Average time per placement: {(end-begin)/num_placements} seconds")
    ###
    if alpha != -1:
        new_jplace_path = jplace_path.split(".jplace")[0]+f"_filtered_{alpha}.jplace"
        with open(new_jplace_path,"w") as fjplace:
            json.dump(jd, fjplace, indent=4)
        filtered_reads_path = jplace_path.split(".jplace")[0]+f"_filtered_{alpha}_reads.txt"
        with open(filtered_reads_path,"w") as ftxt:
            ftxt.write('\n'.join(filtered_reads))

'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input","-i",type=str,required=True)
    parser.add_argument("--output","-o",type=str,required=True)
    #parser.add_argument("--tree","-t",type=str,required=False)
    parser.add_argument("--normalize","-n",action="store_true",required=False,default=True)
    parser.add_argument("--rooted_tree","-r",action="store_true",required=False,default=False)
    #parser.add_argument("--parallelize","-p",action="store_true",required=False)
    parser.add_argument("--random_trials","-rt",type=int,required=False,default=10000)
    parser.add_argument("--random_placements","-rp",type=int,required=False,default=100)
    
    args = parser.parse_args()
    input_jplace = args.input
    #tree = args.tree
    #tree = ts.read_tree_newick(args.tree) if args.tree != None else None
    output = args.output
    random_trials = args.random_trials
    random_placements = args.random_placements
    rooted = args.rooted_tree
    normalize = args.normalize
    #uncertainty_score(input_jplace,output,tree)
    uncertainty_score(input_jplace,output,trials=random_trials,random_placements=random_placements,rooted_tree=rooted, normalize=normalize)
'''
