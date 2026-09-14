#include <algorithm>
#include <string>
#include <map>
#include <vector>
#include <cfloat>
#include <cmath>
#include "compact_tree.h"
#include "CTree_utils.h"
#include "omp.h"

using namespace std;

vector<string> split_string (string s, char  delim){
	stringstream ss(s);
	string split;
	vector<string> splits;
	while(getline(ss,split,delim))
		splits.push_back(split);
	return splits;
}

map<string, CT_NODE_T> label_to_node (compact_tree tree){
	map<string, CT_NODE_T> lbl_to_node;
	for (auto it = tree.levelorder_begin(); it !=  tree.levelorder_end(); ++it){
		lbl_to_node.emplace(tree.get_label(*it),*it);
	}
	return lbl_to_node;
}

float get_placement_error(CT_NODE_T u, CT_NODE_T v, compact_tree tree){
	if (u == v)
		return 0.0;
	else if (u == tree.get_parent(v))
		return tree.get_edge_length(v);
	else if (v == tree.get_parent(u))
		return tree.get_edge_length(u);
	map<CT_NODE_T,float> u_dist; map<CT_NODE_T, float> v_dist;
	u_dist[u] = 0.0; v_dist[v] = 0.0;
	CT_NODE_T curr_node = u;
	CT_NODE_T parent = tree.get_parent(u); 
	while (parent != NULL_NODE){
		u_dist[parent] = u_dist[curr_node] + tree.get_edge_length(curr_node);
		if (parent == v)
			return u_dist[v] - u_dist[tree.get_parent(u)];
		curr_node = parent;
		parent = tree.get_parent(parent);
	}
	auto it = u_dist.find(v);
	if (it != u_dist.end())
		return u_dist[v] - u_dist[tree.get_parent(u)];
	curr_node = v;
	parent = tree.get_parent(v);
	while (parent != NULL_NODE){
		v_dist[parent] = v_dist[curr_node] + tree.get_edge_length(curr_node);
		it = v_dist.find(u);
		if (it != v_dist.end())
			return v_dist[u] - v_dist[tree.get_parent(v)];
		it = u_dist.find(parent);
		if (it != u_dist.end())
			return u_dist[parent] + v_dist[parent] - u_dist[tree.get_parent(u)] - v_dist[tree.get_parent(v)];
		curr_node = parent;
		parent = tree.get_parent(parent);
	}
	return -1.0; //error management
}
