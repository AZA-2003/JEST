#ifndef CTREEUTILS_H
#define CTREEUTILS_H

#include <algorithm>
#include <string>
#include <map>
#include <vector>
#include <cfloat>
#include <cmath>
#include "compact_tree.h"

using namespace std;

struct placement_obj {
	vector<string> n;
	vector<vector<float>> p;
};

vector<string> split_string (string s, char delim);

map<string, CT_NODE_T> label_to_node(compact_tree tree);

//vector<float> get_random_placements_uncertainty (compact_tree tree, size_t trials, size_t random_placements, size_t num_threads);

float get_placement_error(CT_NODE_T u, CT_NODE_T v, compact_tree tree); 

#endif
