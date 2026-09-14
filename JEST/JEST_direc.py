from . import _CTree as CTree
from .JEST_file import uncertainty_score_file
from multiprocessing import Pool
import json
import os
import re
import time


def uncertainty_score_direc(jplace_dir: str,
                            dest_dir: str,
                            tree_path: str,
                            filter_queries: bool = False,
                            alpha: float = 0.05,
                            random_placements: int = 100,
                            trials: int = 10000,
                            num_threads: int = 32,
                            num_procs: int = 1):


    random_uncertainties = list(CTree.get_random_placements_uncertainty(tree_path,trials,random_placements,num_threads,True))
    rp_mean = sum(random_uncertainties)/len(random_uncertainties)
    rp_std = sum([(r-rp_mean)*(r-rp_mean) for r in random_uncertainties])/len(random_uncertainties)
    
   
    direc = [(str(os.path.join(jplace_dir,d)),dest_dir,tree_path,filter_queries,rp_mean,rp_std,alpha,random_placements,trials,num_threads) 
            for d in list(os.listdir(jplace_dir))]

    with Pool(num_procs) as p:
        p.starmap_async(uncertainty_score_file,direc)

    
def placement_consensus_direc():
    pass

def gene_consensus_direc():
    pass
