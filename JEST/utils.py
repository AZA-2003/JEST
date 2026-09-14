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

NO_FIELDS = "WARNING: No fields descriptor found in jplace, now resorting to default jplace format"
ERROR_FILTER = "ERROR: Filter percentile must be between 0 and 1 (set filter_queries to False if you want no filter)!"
ERROR_RANDOM = "ERROR: Random placement parameters must be non-zero positive integers!"
ERROR_CONSENSUS_EDGE_ERROR = "ERROR: Consensus Error chosen but no ground truth label was provided"
ERROR_GAMMA = "ERROR: Gamma  must be a non-zero positive integer!"
#ERROR_METRIC = "ERROR: Metric must either be 'edge error', 'branch length' or 'both'!"

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
   
