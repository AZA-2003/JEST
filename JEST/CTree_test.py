import _CTree as CTree
import os
import json
import time
import numpy as np
import treeswift as ts


def python_based_uncertainty(tree_string,tree_path,placements, trials, random_placements):
    def random_placements_uncertainty(tree, trials, random_placements):
        uncertainties = []
        lbl_to_node = tree.label_to_node("all")
        for __ in range(trials):
            tree_nodes = np.array(list(tree.labels()))
            placements = np.random.choice(tree_nodes,size=random_placements,replace=True)
            ref_placement = np.random.choice(placements,1)[0]
            ref_placement_idx = list(placements).index(ref_placement)
            ref_placement = lbl_to_node[ref_placement]
            #node_placements = [lbl_to_node[p] for p in placements]
            ref_distance = [np.abs(tree.distance_between(lbl_to_node[p],ref_placement)) for p in placements]
            uncertainties.append(sum(ref_distance)/random_placements)
        return uncertainties
    edge_num, l_ratio = 0,4
    scores = []
    tree = ts.read_tree_newick(tree_path)
    lbl_to_node = tree.label_to_node("all")
    rp_uncertainties = random_placements_uncertainty(tree, trials, random_placements)
    rp_mean = np.mean(rp_uncertainties)
    rp_std = np.std(rp_uncertainties)
    rp_uncertainties = (rp_uncertainties-rp_mean)/(rp_std+1e-8)
    idx = 0 
    for placement in placements:
        p_l_ratio = []
        p_lbl = []
        likelihood_str = ""
        ref_distance_str = ""
        for p_idx in range(len(placement['p'])):
            edge = placement['p'][p_idx][edge_num]
            ixe = tree_string.find("{"+f"{edge}"+"}")
            ixs = max((
                (tree_string[:ixe].rfind(",")),
                (tree_string[:ixe].rfind("(")),
                (tree_string[:ixe].rfind(")")),
                    ))
            likelihood = float(placement['p'][p_idx][l_ratio])
            #if idx == 33:
            #    likelihood_str += (str(likelihood)+"\t")
            lbl_placement  = tree_string[ixs+1:ixe].split(":")[0]
            p_l_ratio.append(likelihood)
            p_lbl.append(lbl_placement)
        p_l_ratio = np.array(p_l_ratio)
        p_l_ratio /= (np.sum(p_l_ratio))
        node_placements = [lbl_to_node[p] for i,p in enumerate(p_lbl)]
        ref_placement_idx = np.argmax(p_l_ratio)
        ref_placement = p_lbl[ref_placement_idx]
        ref_placement = lbl_to_node[ref_placement]
        ref_distance = [np.abs(tree.distance_between(p,ref_placement)) for p in node_placements]
        #if idx  == 33:
        #    for i, d in enumerate(ref_distance):
        #        ref_distance_str += (str(d)+"\t")
        weighted_score = sum([p_l_ratio[i]*ref_distance[i] for i in range(len(p_l_ratio))])
        #if idx == 33:
        #    print(likelihood_str+str(ref_placement)+"\t"+ref_distance_str+str(weighted_score))
        scores.append(float((weighted_score-rp_mean)/(rp_std+1e-8)))
        #print(str(weighted_score)+"\t"+str(scores[-1]))
        idx +=1
    return scores
#for i in range(1,6):
cpp_times_u = []
cpp_times_cg = []
cpp_times_cp = []
py_times = []
speedups = []
n = 1
direc = "Minderoo/placements/consensus_tool_test/archaea_markers_subsample_100"
for f in os.listdir(f"{direc}"):
    gt = f.split(".jplace")[0]
    print(f"test{n} ############## {gt}")
    with open(f"{direc}/{f}","r") as ff:
        jfile = json.load(ff)
    
    tree_string = jfile["tree"]
    placements = jfile["placements"]

    begin = time.time()
    Cpp_scores = list(CTree.get_raw_uncertainty(tree_string, "Minderoo/data/markers_all_genomes/pruned_labeled.nwk", placements, 100,5, 64))
    end = time.time()
    print(f"Cpp uncertainty: {end-begin}")
    cpp_time = end-begin
    cpp_times_u.append(cpp_time)
    
    '''
    begin = time.time()
    Py_scores = python_based_uncertainty(tree_string, "test_tree.nwk", placements, 100, 5)
    end = time.time()
    print(f"Py uncertainty: {end-begin}")
    py_time = end-begin
    py_times.append(py_time)
    
    speedups.append(round(py_time/cpp_time, 3))
    print(f"CPP speed-up = {round(py_time/cpp_time, 3)}")
    '''
    
    
    begin = time.time()
    Cpp_consensus_p = list(CTree.placement_consensus(tree_string, "Minderoo/data/markers_all_genomes/pruned_labeled.nwk", placements, 10.0, gt, 64))
    end = time.time()
    print(f"Cpp placement consensus: {end-begin}")
    cpp_time = end-begin
    cpp_times_cp.append(cpp_time)
     
    begin = time.time()
    Cpp_consensus_g = list(CTree.gene_consensus(tree_string, "Minderoo/data/markers_all_genomes/pruned_labeled.nwk", placements, 10.0, gt, 64))
    end = time.time()
    print(f"Cpp gene consensus: {end-begin}")
    cpp_time = end-begin
    cpp_times_cg.append(cpp_time)

    #print(Cpp_scores[:5], Py_scores[:5])
    '''
    assert len(Cpp_scores) == len(Py_scores)

    for i in range(len(Cpp_scores)):
        if (abs(Cpp_scores[i] - Py_scores[i]) > 1):
            print(f"{i}\t{Cpp_scores[i]}\t{Py_scores[i]}\t{Cpp_scores[i] - Py_scores[i]}")
    '''
    n +=1
    '''
    if n > 5:
        break
    '''
print("Success!")
print(f"average CPP time for uncertainty: {round(sum(cpp_times_u)/len(cpp_times_u),4)}")
print(f"average CPP time for placement consensus: {round(sum(cpp_times_cp)/len(cpp_times_cp),4)}")
print(f"average CPP time for placement consensus: {round(sum(cpp_times_cg)/len(cpp_times_cg),4)}")
#print(f"average Python time: {round(sum(py_times)/len(py_times),4)}")
#print(f"average speedup: {round(sum(speedups)/len(speedups),4)}")
