import os
import errno
import sys
import treeswift as ts
import gzip
import json
import re
import argparse
import numpy as np
from scipy import stats


NO_FIELDS= "WARNING: No fields descriptor found in jplace, now resorting to default jplace format"

def label_internal_nodes(tree_str):
    ref_str = tree_str[:]
    idx = 0
    while True:
        tree_str = ref_str.replace("):",f")E{idx}:",1)
        if tree_str == ref_str:
            break
        ref_str = tree_str[:]
        idx += 1
    tree_str = ref_str.replace(");",f")E{idx};",1)
    return tree_str
    '''
    idx = 0
    for node in tree.traverse_internal():
        node.label = "E"+str(idx)
        idx += 1
    return tree
    '''

'''
takes in a path for jplace and returns a dictionary
works on both zipped and unzipped jplace files
'''
def read_jplace(jplace_path):
    try:
        f_name = jplace_path.split("/")[-1]
        if ".gz"in f_name:
            with gzip.open(jplace_path,"rt") as f:
                jd = json.load(f)
        else:
            with open(jplace_path,"r") as f:
                jd = json.load(f)
        return jd
    except FileNotFoundError as E:
        raise E(errno.ENOENT, os.strerror(errno.ENOENT), jplace_path)
'''
Generates a number of trials where random placements are made
'''
def random_placement_score(tree, trials, random_placements):
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
    norm_uncertainties = np.array(uncertainties) / (sum(uncertainties)/trials)
    ## TODO Add a verbose feature to print out stats
    '''
    print(f"1 percentile is {np.percentile(norm_uncertainties,1)}")
    print(f"0.1 percentile is {np.percentile(norm_uncertainties,0.1)}")
    print(f"75 percentile is {np.percentile(norm_uncertainties,75)}")
    print(f"90 percentile is {np.percentile(norm_uncertainties,90)}")
    print(f"0.5 lies on {stats.percentileofscore(norm_uncertainties,0.5)} percentile")
    print(f"mean of random distribution is {(sum(uncertainties)/trials)}")
    '''
    #norm_uncertainties = np.array(uncertainties)
    #print(norm_uncertainties)
    #norm_uncertainties /= np.median(uncertainties)
    #norm_uncertainties /= np.mean(uncertainties)
    return norm_uncertainties, (sum(uncertainties)/trials)

## CREATE A MULTITHREADED VERSION OF uncertainty_score


'''
takes in a path for jplace and an output destination file to write the uncertainty score of all 
query in the jplace file
'''
def uncertainty_score(jplace_path,dest_path, normalize=True,
                        trials=10000,random_placements=100, rooted=False):
    jd = read_jplace(jplace_path)
    jtree = jd["tree"]
    #jtree = tree.newick()
    try:
        jtree = label_internal_nodes(jtree)
        jtree_r = re.sub(r"{[0-9]+}","",jtree)
        tree = ts.read_tree_newick(jtree_r)            
    except Exception as E:
        print("Error!")
        exit()
    
    normalizer = 1.0
    if normalize:
        random_pl,normalizer = random_placement_score(tree, trials, random_placements)
        #np.save("uce_random_dist.npy",random_pl)
        random_dist = stats.ecdf(random_pl)
        #normalizer = np.mean(random_pl)
        #print(normalizer)
    try:
        fields = jd["fields"]
        edge_num = fields.index("edge_num")
        l_ratio = fields.index("like_weight_ratio")
    except Exception as E:
        print(NO_FIELDS)
        edge_num = 0
        l_ratio = 2
    lbl_to_node = tree.label_to_node("all")
    with open(dest_path,"w") as f:
        f.write(f"name\tuncertainty\tuncertainty_ratio\tpercentile\n")
        ## iterating over all queries
        for placement in jd["placements"]:
            name = placement['n'][0]
            p_l_ratio = []
            p_edge_num =[]
            p_lbl = []
            ## gather the likeihood and edge number
            if len(placement['p']) == 0:
                f.write(f"{name}\tNaN\tNaN\tNaN\n")
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
            if rooted:
                ref_placement = tree.mrca(p_lbl)
                ref_distance = [np.abs(tree.distance_between(p,ref_placement)) for p in node_placements]
                weighted_score = sum([p_l_ratio[i]*ref_distance[i] for i in range(len(p_l_ratio))])
            else:
                ref_placement_idx = np.argmax(p_l_ratio)
                ref_placement = p_lbl[ref_placement_idx]
                ref_placement = lbl_to_node[ref_placement]
                ref_distance = [np.abs(tree.distance_between(p,ref_placement)) for p in node_placements]
                weighted_score = sum([p_l_ratio[i]*ref_distance[i] for i in range(len(p_l_ratio))])/(1-p_l_ratio[ref_placement_idx]) if p_l_ratio[ref_placement_idx] != 1.0 else 0.0
            
            weighted_score_norm = weighted_score /normalizer
            #proportion = stats.percentileofscore(random_pl,weighted_score)
            proportion = stats.percentileofscore(random_pl,weighted_score_norm)/100

            f.write(f"{name}\t{weighted_score}\t{weighted_score_norm}\t{proportion}\n")

'''
measures distance between placement and ground truth
either via edge error or by branch length on a reference tree
'''
def placement_error(jplace_path: str, dest_path: str,
        ground_truth: str, ref_tree_path: str, metric: str):
    
    NaN = float("nan")
    if metric not in ["edge error", "branch length", "both"]:
        raise Exception("Invalid metric! Metric must either be 'edge error', 'branch length' or 'both'!")
    '''
    sub function for distance calculation
    '''
    def get_distance(u,v):
        if u == v:
            return 0.0
        elif u == v.parent:
            return v.edge_length
        elif v == u.parent:
            return u.edge_length
        u_dists = {u: 0.0}
        v_dists = {v: 0.0}
        ## u traversal upwards
        c = u
        p = u.parent  # u traversal
        while p is not None:
            u_dists[p] = u_dists[c]
            if c.edge_length is not None:
                u_dists[p] += c.edge_length #accumulate edge lengths from u during traversal
            c = p
            p = p.parent
        ## case where v is simply an ancestor of u (parent of parent)
        if v in u_dists:
            return u_dists[v] - u_dists[u.parent]
        ## v traversal upwards
        c = v
        p = v.parent  # v traversal
        while p is not None:
            v_dists[p] = v_dists[c]
            if c.edge_length is not None:
                v_dists[p] += c.edge_length #accumulate edge lengths from v during traversal
            ## case where the ancestors of u and v meet 
            if p in u_dists:
                return u_dists[p] + v_dists[p] - u_dists[u.parent] - v_dists[v.parent]#distance from u to that ancestor + from v to that ancestor
            c = p
            p = p.parent


    jd = read_jplace(jplace_path)
    placement_tree = jd["tree"]

    reference_tree = ts.read_tree_newick(ref_tree_path)
    ground_truth_node = reference_tree.find_node(ground_truth, leaves=True, internal=True)
    lbl_to_nd =  reference_tree.label_to_node("all")
   

    reference_tree_norm = ts.read_tree_newick(ref_tree_path)
    ground_truth_node_norm = reference_tree.find_node(ground_truth, leaves=True, internal=True)
    lbl_to_nd_norm =  reference_tree.label_to_node("all")


    for nd in reference_tree_norm.traverse_postorder():
        nd.set_edge_length(1)
    with open(f"{dest_path}/eval.txt","w") as f:
        for placements in jd["placements"]:
            edge_error = "-"
            branch_length = "-"
            read_id = placement["n"][0]
            if True: #for now..
                if len(placement["p"]) == 0:
                    print(
                            f"{sys.argv[2]}\t{rid}\tNaN\tNaN\tNaN\tNaN"
                        )
                    continue
                for p_idx in range(len(placement['p'])):
                    ixe = placement_tree.find("{" + f"{placement['p'][idx][0]}" + "}")
                    ixs = max(
                                (placement_tree[:ixe].rfind(","),
                                placement_tree[:ixe].rfind("("),
                                placement_tree[:ixe].rfind(")"),)
                            )
                    likelihood = float(placement['p'][p_idx][4])
                    lbl_placement, blen = placement_tree[ixs + 1 : ixe].split(":")
                    if metric in ["edge error","both"]:
                        if lbl_placement == ground_truth:
                            edge_error = 0.0
                        else:
                            edge_error = distance_between(ground_truth_node_norm, lbl_to_nd_norm[lbl_placement])
                    if metric in ["branch length","both"]:
                        if lbl_placement == ground_truth:
                            branch_length = 0.0
                        else:
                            branch_length = distance_between(ground_truth_node, lbl_to_nd[lbl_placement])
                    f.write(f"{read_id}\t{ground_truth}\t{lbl_placement}\t{round(likelihood,6)}\t{edge_error}\t{round(branch_length,6)}")

'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input","-i",type=str,required=True)
    parser.add_argument("--output","-o",type=str,required=True)
    #parser.add_argument("--tree","-t",type=str,required=False)
    parser.add_argument("--normalize","-n",action="store_true",required=False,default=True)
    parser.add_argument("--rooted","-r",action="store_true",required=False,default=False)
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
    rooted = args.rooted
    normalize = args.normalize
    #uncertainty_score(input_jplace,output,tree)
    uncertainty_score(input_jplace,output,trials=random_trials,random_placements=random_placements,rooted=rooted, normalize=normalize)
'''
