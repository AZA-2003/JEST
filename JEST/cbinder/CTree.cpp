#include "compact_tree.h"
#include "CTree_utils.h"
#include <string>
#include <map>
#include <unordered_set>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <regex>
#include <tuple>
#include <cfloat>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cassert>
#include <omp.h>

using namespace std;
namespace fs = std::filesystem;

//TODO modify function signature to take in vector<vector<float>> for in-place modification of distal and pendant length returning nothing..
vector<vector<float>> find_length(string jtree, string tree_string, vector<placement_obj>& placements, vector<string> consensus, 
				size_t num_threads, size_t lwr_idx, size_t en_idx){

	vector<float> distal_lengths(placements.size());
	vector<float> pendant_lengths(placements.size());
	distal_lengths.reserve(placements.size());
	pendant_lengths.reserve(placements.size());
	compact_tree tree;
	tree = compact_tree(tree_string,false,true,true,0);
	map<string, CT_NODE_T> lbl_to_node = label_to_node(tree);
	size_t ol_threads = min(num_threads,placements.size());
	omp_set_nested(1);
	omp_set_num_threads(ol_threads);
	#pragma omp parallel for
	for (size_t idx = 0; idx < placements.size(); idx++){
		placement_obj & placement = placements[idx];
		vector<string> p_lbl(placement.p.size());
		vector<float> p_l_ratio(placement.p.size());
		//omp_set_num_threads(min(num_threads,placement.p.size()));
		//#pragma omp parallel for
		for (size_t p_idx = 0; p_idx < placement.p.size(); p_idx++){
			int edge = placement.p[p_idx][en_idx];
			float likelihood = placement.p[p_idx][lwr_idx];
			int ixe = jtree.find("{"+to_string(edge)+"}");
			string tree_substring = jtree.substr(0,ixe);
			int ixs = max({(int)tree_substring.rfind(','),
						(int)tree_substring.rfind('('),
						(int)tree_substring.rfind(')')});
			vector<string> jtree_split = split_string(jtree.substr(ixs+1, ixe-ixs-1), ':');
			string lbl_placement = jtree_split[0];
			p_lbl[p_idx] = lbl_placement;
			p_l_ratio[p_idx] = likelihood;
		}
		float p_l_ratio_sum = accumulate(p_l_ratio.begin(), p_l_ratio.end(), 0.0);
		if (p_l_ratio_sum == 0){
			distal_lengths[idx] = NAN;
			continue;
		}
		// nice trick to normalize the likelihood weight ratio
		auto normalize_p_l_ratio = [& p_l_ratio_sum](float &n){n = n/p_l_ratio_sum;};
		for_each(p_l_ratio.begin(), p_l_ratio.end(), normalize_p_l_ratio);
		
		float nabla = 0.01;
	   	string consensus_lbl = consensus[idx];
		CT_NODE_T consensus_node = lbl_to_node[consensus_lbl];
		float consensus_blen = tree.get_edge_length(consensus_node);
		float distal = ((float)rand()) / RAND_MAX * consensus_blen;
		float pendant = 0.0;
		float pendant_norm = 0.0;
		//cout << "Branch Length of Consensus: "<< consensus_blen<<"\n";
		float diff_consensus = 0.0;
		for (size_t pl_idx = 0; pl_idx < p_lbl.size(); pl_idx++){
			CT_NODE_T p_node = lbl_to_node[p_lbl[pl_idx]];
			string ref_lbl = p_lbl[pl_idx];
			while (ref_lbl != tree.get_label(tree.get_root())){
				if (ref_lbl == consensus_lbl){
					pendant += tree.calc_dist(p_node, consensus_node)*p_l_ratio[pl_idx];
					//pendant_norm += p_l_ratio[pl_idx];
					pendant_norm += 1.0;
					diff_consensus += p_l_ratio[pl_idx];
					break;
				}
				ref_lbl = tree.get_label(tree.get_parent(lbl_to_node[ref_lbl]));
			}
			if (ref_lbl == tree.get_label(tree.get_root())){
				diff_consensus -= p_l_ratio[pl_idx];
			}
		}
		pendant/= pendant_norm;
		//cout << "Pendant length is " << pendant << "\n";
		//cout << "Performing Optimization Run!\n";
		for (size_t iter = 0; iter < 20; iter++){
			distal = min(consensus_blen, max((float)0.0, distal-nabla*diff_consensus));
		}
		pendant_lengths[idx] = pendant;
		distal_lengths[idx] = distal;
	}
	return {distal_lengths, pendant_lengths};
}




void placement_error(string jtree, string tree_string, vector<placement_obj>& placements,string reference, size_t num_threads, bool edge_error, string dest_path, vector<string> fields){
	
	ofstream outFile(dest_path, ios::out);
	size_t en_idx = distance(fields.begin(), find(fields.begin(), fields.end(), "edge_num"));
	size_t lwr_idx = distance(fields.begin(), find(fields.begin(), fields.end(), "like_weight_ratio"));

	compact_tree tree;

	tree = compact_tree(tree_string,true,true,true,0);

	map<string, CT_NODE_T> lbl_to_node = label_to_node(tree);
	if (edge_error == true){
		for (CT_NODE_T node = 0; node < tree.get_num_nodes(); ++node) tree.set_edge_length(node,1.0);
		outFile << "name\treference\tedge error\n";
	}
	else{
		outFile << "name\treference\tbranch length error\n";
	}
	size_t ol_threads = min(num_threads,placements.size());
	omp_set_nested(1);
	omp_set_num_threads(ol_threads);
	#pragma omp parallel for
	for (size_t idx = 0; idx < placements.size(); idx++){
		placement_obj & placement = placements[idx];
		string lbl_placement;
		float max_likelihood = 0.0;
		if (placement.p.size() == 0){
			outFile << placement.n[0] << "\t" << reference[idx] << "\t" << "N/A\n";
			continue;
		}
		float weighted_error = 0.0;
		for (size_t p_idx = 0; p_idx < placement.p.size(); p_idx++){
			int edge = placement.p[p_idx][en_idx];
			int ixe = jtree.find("{"+to_string(edge)+"}");
			string tree_substring = jtree.substr(0,ixe);
			int ixs = max({(int)tree_substring.rfind(','),
						(int)tree_substring.rfind('('),
						(int)tree_substring.rfind(')')});
			float likelihood = placement.p[p_idx][lwr_idx];
			vector<string> jtree_split = split_string(jtree.substr(ixs+1, ixe-ixs-1), ':');
			lbl_placement = jtree_split[0];
			weighted_error += likelihood * get_placement_error(lbl_to_node[lbl_placement], lbl_to_node[reference],tree);
		}

		#pragma omp critical
		outFile << placement.n[0] << "\t" << weighted_error << "\n";
		
	}
	outFile.close();
}

// TODO will add some error checks for the list to correspond to the placement size and ensure proper swig translation
void placement_error_multilabel(string jtree, string tree_string, vector<placement_obj>& placements, vector<string> reference, size_t num_threads, bool edge_error, string dest_path, vector<string> fields){
	
	ofstream outFile(dest_path, ios::out);
	compact_tree tree;
	size_t en_idx = distance(fields.begin(), find(fields.begin(), fields.end(), "edge_num"));
	size_t lwr_idx = distance(fields.begin(), find(fields.begin(), fields.end(), "like_weight_ratio"));

	tree = compact_tree(tree_string,false,true,true,0);
	map<string, CT_NODE_T> lbl_to_node = label_to_node(tree);
	if (edge_error == true){
		for (CT_NODE_T node = 0; node < tree.get_num_nodes(); ++node) tree.set_edge_length(node,1.0);
		outFile << "name\treference\tedge error\n";
	}
	else{
		outFile << "name\treference\tbranch length error\n";
	}
	size_t ol_threads = min(num_threads,placements.size());
	omp_set_nested(1);
	omp_set_num_threads(ol_threads);
	#pragma omp parallel for
	for (size_t idx = 0; idx < placements.size(); idx++){
		placement_obj & placement = placements[idx];
		string lbl_placement;
		float max_likelihood = 0.0;
		if (placement.p.size() == 0){
			outFile << placement.n[0] << "\t" << reference[idx] << "\t" << "N/A\n";
			continue;
		}
		float weighted_error = 0.0;
		for (size_t p_idx = 0; p_idx < placement.p.size(); p_idx++){
			int edge = placement.p[p_idx][en_idx];
			int ixe = jtree.find("{"+to_string(edge)+"}");
			string tree_substring = jtree.substr(0,ixe);
			int ixs = max({(int)tree_substring.rfind(','),
						(int)tree_substring.rfind('('),
						(int)tree_substring.rfind(')')});
			float likelihood = placement.p[p_idx][lwr_idx];
			vector<string> jtree_split = split_string(jtree.substr(ixs+1, ixe-ixs-1), ':');
			lbl_placement = jtree_split[0];
			weighted_error += likelihood * get_placement_error(lbl_to_node[lbl_placement], lbl_to_node[reference[idx]],tree);
		}
		#pragma omp critical
		outFile << placement.n[0] << "\t" << reference[idx] << "\t" << weighted_error << "\n";
		
	}
	outFile.close();
}


vector<float> get_random_placements_uncertainty (string tree_string, size_t trials, size_t random_placements, size_t num_threads, bool is_path){
	compact_tree tree;
	if (is_path)
		tree = compact_tree(tree_string,true,true,true,0);
	else
		tree = compact_tree(tree_string,false,true,true,0);

	map<string, CT_NODE_T> lbl_to_node = label_to_node(tree);
	vector<string> tree_labels = tree.get_labels();
	vector<float> uncertainties(trials);
	uncertainties.reserve(trials);
	CT_NODE_T ref_placement;
    omp_set_num_threads(num_threads);
	#pragma omp parallel for
	for (size_t i = 0; i < trials; i++){
		vector<string> placements;
		vector<float> ref_distances;	
		sample(tree_labels.begin(), tree_labels.end(), back_inserter(placements), random_placements,mt19937{random_device{}()});
		ref_placement = lbl_to_node[placements[rand()%random_placements]];
		for (size_t j = 0; j < random_placements; j++){
			ref_distances.push_back(abs(tree.calc_dist(ref_placement,lbl_to_node[placements[j]])));
		}
		uncertainties[i] = accumulate(ref_distances.begin(), ref_distances.end(),0.0)/random_placements;
	}
	//sort(uncertainties.begin(), uncertainties.end());
	return uncertainties;
}
map<string,float> get_raw_uncertainty(string jtree, string tree_string, vector<placement_obj>& placements, 
								size_t num_threads, bool is_path,
								size_t lwr_idx, size_t en_idx, bool rooted){
	/** placment should consist of n and p where 
	 * n is the name of the placement 
	 * p is the placement info of all possible 
	 * placements 
	 * Assume single dictionary **/
	
	/** assume default set-up where 
	 * edge length: index 0
	 * pendant length: index 1
	 * distal length: index 2
	 * likelihood: index 3
	 * likelihood weight ratio: index 4 
	 * distance: index 5
	 * **/
	compact_tree tree;
	if (is_path)
		tree = compact_tree(tree_string,true,true,true,0);
	else
		tree = compact_tree(tree_string.c_str(),false,true,true,0);
	
	// To be removed
	//for (CT_NODE_T node = 0; node < tree.get_num_nodes(); ++node) tree.set_edge_length(node,1.0);
	map<string, CT_NODE_T> lbl_to_nd = label_to_node(tree);
	vector<pair<string,float>> uncertainty_scores(placements.size());
	size_t ol_threads = min(num_threads,placements.size());
	omp_set_nested(1);
	omp_set_num_threads(ol_threads);
	#pragma omp parallel for
	for (size_t idx = 0; idx < placements.size(); idx++){
		placement_obj & placement = placements[idx];
		vector<string> p_lbl(placement.p.size());
		vector<float> p_l_ratio(placement.p.size());
		//omp_set_num_threads(min(num_threads,placement.p.size()));
		//#pragma omp parallel for
		for (size_t p_idx = 0; p_idx < placement.p.size(); p_idx++){
			int edge = placement.p[p_idx][en_idx];
			float likelihood = placement.p[p_idx][lwr_idx];
			int ixe = jtree.find("{"+to_string(edge)+"}");
			string tree_substring = jtree.substr(0,ixe);
			int ixs = max({(int)tree_substring.rfind(','),
						(int)tree_substring.rfind('('),
						(int)tree_substring.rfind(')')});
			vector<string> jtree_split = split_string(jtree.substr(ixs+1, ixe-ixs-1), ':');
			string lbl_placement = jtree_split[0];
			p_lbl[p_idx] = lbl_placement;
			p_l_ratio[p_idx] = likelihood;
		}
		float p_l_ratio_sum = accumulate(p_l_ratio.begin(), p_l_ratio.end(), 0.0);
		if (p_l_ratio_sum == 0){
			uncertainty_scores[idx] = {placement.n[0], NAN};
			continue;
		}
		// nice trick to normalize the likelihood weight ratio
		auto normalize_p_l_ratio = [& p_l_ratio_sum](float &n){n = n/p_l_ratio_sum;};
		for_each(p_l_ratio.begin(), p_l_ratio.end(), normalize_p_l_ratio);
		vector<CT_NODE_T> node_placements;
		for (size_t i =0; i < p_lbl.size(); i++)
			node_placements.push_back(lbl_to_nd[p_lbl[i]]);

		// step 2: calculate raw uncertainty
		
		// Rooted case
		CT_NODE_T ref_placement;
		if (rooted == true){
			auto  get_p_lbl_nodes = [& lbl_to_nd](string lbl)-> CT_NODE_T {return lbl_to_nd[lbl];};
			unordered_set<CT_NODE_T> p_nodes = {};
			for (size_t k = 0; k < p_lbl.size(); k++)
				p_nodes.insert(get_p_lbl_nodes(p_lbl[k]));
			ref_placement = tree.find_mrca(p_nodes);
		}	
		else{
		// Unrooted case
			size_t ref_placement_idx = distance(p_l_ratio.begin(), max_element(p_l_ratio.begin(), p_l_ratio.end()));
			//cout << p_lbl[ref_placement_idx] <<"\t";
			ref_placement = lbl_to_nd[p_lbl[ref_placement_idx]];
		}
		//cout << tree.get_label(ref_placement) << "\t";
		//cout << node_placements.size() << "\t";
		float weighted_score = 0.0;
		for (size_t j = 0; j < node_placements.size(); j++){
			//cout << node_placements[j] << "\t";
			//cout << tree.calc_dist(ref_placement, node_placements[j])  << "\t";
			weighted_score += abs(tree.calc_dist(ref_placement, node_placements[j])) * p_l_ratio[j];
		}
		//TODO add a small tweak where score is then divided by (1-ref_placement_likelihood)
		/**
		if (p_l_ratio[ref_placement_idx] != 1.0)
			weighted_score /= (1-p_l_ratio[ref_placement_idx]);
		**/
		uncertainty_scores[idx] = {placement.n[0], weighted_score};
	}
	map<string,float> placement_uncertainties(uncertainty_scores.begin(), uncertainty_scores.end());
	return placement_uncertainties;
}

map<string,float> get_uncertainty_pvalue(string jtree, string tree_string, vector<placement_obj>& placements, 
					vector<float> random_dist, 
					size_t num_threads, string dest_path, 
					bool is_path,vector<string> fields, bool rooted){
	
	size_t en_idx = distance(fields.begin(), find(fields.begin(), fields.end(), "edge_num"));
	size_t lwr_idx = distance(fields.begin(), find(fields.begin(), fields.end(), "like_weight_ratio"));
	ofstream outFile(dest_path, ios::out);
	map<string,float> uncertainties = get_raw_uncertainty(jtree,tree_string,placements,num_threads,is_path, lwr_idx, en_idx, rooted);
	float rp_mean = accumulate(random_dist.begin(), random_dist.end(), 0.0) /random_dist.size();
	auto mean_center_square = [&rp_mean](float& v){v = (v-rp_mean)*(v-rp_mean);};
	vector<float> rp_mean_center(random_dist);
	for_each(rp_mean_center.begin(), rp_mean_center.end(), mean_center_square);
	float rp_std = sqrt(accumulate(rp_mean_center.begin(), rp_mean_center.end(), 0.)/rp_mean_center.size());
	//cout << rp_mean <<"\t"<< rp_std<<"\n";
	auto normalize_rd = [&rp_mean, &rp_std](float&  v) {v = (v-rp_mean)/(rp_std+1e-5f);};
	auto normalize = [&rp_mean, &rp_std](float  v) {return (v-rp_mean)/(rp_std+1e-5f);};
	//auto normalize_rd = [&rp_mean, &rp_std](float&  v) {v = (v/(rp_mean+1e-8f));};
	//auto normalize = [&rp_mean, &rp_std](float  v) {return (v/(rp_mean+1e-8f));};
	//auto get_pvalue = [&rp_mean, &rp_std](float kv) {return 0.5+0.5*erf((kv-rp_mean)*M_SQRT1_2/(rp_std+1e-8f));};
	for_each(random_dist.begin(), random_dist.end(), normalize_rd);
	map<string,float> pval_uncertainties;
	outFile << "name\tuncertainty raw score\tuncertainty normalized\tuncertainty p-value\n";
	setprecision(4);
	for(auto it = uncertainties.begin(); it != uncertainties.end(); ++it){
		//cout << it->second <<"\t"<< normalize(it->second) << "\n";
		pval_uncertainties[it->first] = Percentile(random_dist, normalize(it->second));
		outFile << it->first << "\t" << it->second <<"\t"<< normalize(it->second) <<"\t" << pval_uncertainties[it->first] << "\n";
	}
	outFile.close();
	return pval_uncertainties;
}

map<string, vector<string>> label_children (string tree_string, bool is_path){
	compact_tree tree;
	if (is_path)
		tree = compact_tree(tree_string,true,true,true,0);
	else
		tree = compact_tree(tree_string,false,true,true,0);
	map<string, vector<string>> mapper;
	for (auto it = (tree).levelorder_begin(); it != (tree).levelorder_end(); ++it){
		CT_NODE_T node = *it;
		vector<CT_NODE_T> children = (tree).get_children(node);
		for (int i  = 0; i  < children.size(); i++)
			mapper[(tree).get_label(node)].push_back((tree).get_label(children[i]));
	}
	return mapper;
}

//vector<string> placement_consensus (string jtree, string tree_string, vector<placement_obj>& placements, size_t num_threads, string dest_path, bool get_error, string ground_truth, string gt_tree, size_t lwr_idx, size_t en_idx){
vector<string> get_placement_consensus (string jtree, string tree_string, vector<placement_obj>& placements, size_t num_threads, vector<string> fields){
	
	size_t en_idx = distance(fields.begin(), find(fields.begin(), fields.end(), "edge_num"));
	size_t lwr_idx = distance(fields.begin(), find(fields.begin(), fields.end(), "like_weight_ratio"));
	size_t plen_idx = distance(fields.begin(), find(fields.begin(), fields.end(), "pendant_length"));
	size_t dlen_idx = distance(fields.begin(), find(fields.begin(), fields.end(), "distal_length"));
	size_t l_idx = distance(fields.begin(), find(fields.begin(), fields.end(), "likelihood"));

	vector<string> consensus(placements.size());
	consensus.reserve(placements.size());
	compact_tree tree(tree_string,false,true,true,0);

	map<string, CT_NODE_T> lbl_to_nd = label_to_node(tree);
	omp_set_nested(1);
	size_t ol_threads = min(num_threads,placements.size());
	omp_set_num_threads(ol_threads);
	#pragma omp parallel for shared(tree,lbl_to_nd)
	for (size_t c_idx = 0; c_idx < placements.size(); c_idx++){
		placement_obj & placement = placements[c_idx];
		vector<string> p_lbl;
		vector<float> p_l_ratio;
		// step 1:  gather the node placements and their likelihood ratios
		for (size_t p_idx = 0; p_idx < placement.p.size(); p_idx++){
			int edge = (int)placement.p[p_idx][en_idx];
			int ixe = jtree.find("{"+to_string(edge)+"}");
			string tree_substring = jtree.substr(0,ixe);
			int ixs = max({(int)tree_substring.rfind(','),
						(int)tree_substring.rfind('('),
						(int)tree_substring.rfind(')')});
			vector<string> jtree_split = split_string(jtree.substr(ixs+1, ixe-ixs-1), ':');
			p_lbl.push_back(jtree_split[0]);
			p_l_ratio.push_back(placement.p[p_idx][lwr_idx]);
		}
		float p_l_ratio_sum = accumulate(p_l_ratio.begin(), p_l_ratio.end(), 0.0);
		if (p_l_ratio_sum == 0.0){
			//outFile << placement.n[0] << "\t" << "N/A" << "N/A";
			cout << "Zero Placements found, unable to find a consensus!\n";
			continue;
		}
		// nice trick to normalize the likelihood weight ratio
		auto normalize_p_l_ratio = [& p_l_ratio_sum](float &n){n = n/p_l_ratio_sum;};
		for_each(p_l_ratio.begin(), p_l_ratio.end(), normalize_p_l_ratio);

		// step 2: perform placement consensus
		string jest_label;
		float max_jest_score = 0.0;
		
		if (p_lbl.size() == 1){
			jest_label = p_lbl[0];
		}
		else{
			vector<CT_NODE_T> node_placements;
			for (size_t i =0; i < p_lbl.size(); i++)
				node_placements.push_back(lbl_to_nd[p_lbl[i]]);

			size_t ref_placement_idx = distance(p_l_ratio.begin(), max_element(p_l_ratio.begin(), p_l_ratio.end()));
			CT_NODE_T ref_placement = lbl_to_nd[p_lbl[ref_placement_idx]];
			float weighted_score = 0.0;
			for (size_t j = 0; j < node_placements.size(); j++){
				weighted_score += abs(tree.calc_dist(ref_placement, node_placements[j])) * p_l_ratio[j];
			}
			float w_gamma  = 1/(weighted_score + 1e-8f);
			map<string, float> TP_node, FP_node, FN_node;
			map<string, size_t> count;
			auto f_score = [&TP_node, &FP_node, &FN_node ](string lbl) -> float {
					if(TP_node[lbl] > 0.0) return 2*TP_node[lbl]/(2*TP_node[lbl] + FP_node[lbl] + FN_node[lbl]); 
					else 
						return 0.0;
			};	
			deque<string> Deque;
			for (string p : p_lbl) Deque.push_back(p);

			string carrier_lbl;
			while (Deque.size() != 0){
				carrier_lbl = Deque.front();
				count[carrier_lbl] += 1;
				auto idx = find(p_lbl.begin(), p_lbl.end(), carrier_lbl);
				if (idx != p_lbl.end()){
					size_t p_idx = distance(p_lbl.begin(), idx);	
					TP_node[carrier_lbl] = p_l_ratio[p_idx]*w_gamma;
					FP_node[carrier_lbl] = (1.0-p_l_ratio[p_idx])*w_gamma;
				}else{
					TP_node[carrier_lbl] = 0.0;
					FP_node[carrier_lbl] = w_gamma;	
				}
				if (!tree.is_leaf(lbl_to_nd[carrier_lbl])){
					for (CT_NODE_T child : tree.get_children(lbl_to_nd[carrier_lbl])){
						string child_lbl = tree.get_label(child);
						TP_node[carrier_lbl] += TP_node[child_lbl]*exp(-w_gamma*tree.calc_dist(lbl_to_nd[carrier_lbl],child));
						FP_node[carrier_lbl] += FP_node[child_lbl]*exp(-w_gamma*tree.calc_dist(lbl_to_nd[carrier_lbl],child));
					}
				}
				if (count[carrier_lbl] == p_lbl.size())
					break;
				else if (tree.get_parent(lbl_to_nd[carrier_lbl]) != NULL_NODE)
					Deque.push_back(tree.get_label(tree.get_parent(lbl_to_nd[carrier_lbl])));
				Deque.pop_front();		
			}
			FN_node[carrier_lbl] = 0.0;
			jest_label = carrier_lbl;
			//cout << carrier_lbl << "\t" << tree.get_label(tree.get_root()) <<"\n";
			if (carrier_lbl == tree.get_label(tree.get_root()))
				max_jest_score = 0.0;
			else
				max_jest_score = f_score(carrier_lbl);
			Deque.clear();
			for (CT_NODE_T child : tree.get_children(lbl_to_nd[carrier_lbl])){
				string child_lbl = tree.get_label(child);
				Deque.push_back(child_lbl);
			}
			while (Deque.size() != 0){
				carrier_lbl = Deque.front();
				CT_NODE_T parent = tree.get_parent(lbl_to_nd[carrier_lbl]);
				string parent_lbl = tree.get_label(parent);
				FN_node[carrier_lbl] = (FN_node[parent_lbl] + FP_node[parent_lbl] -
								(FP_node[carrier_lbl] * exp(-w_gamma * tree.calc_dist(parent,lbl_to_nd[carrier_lbl])))) *
								exp(-w_gamma * tree.calc_dist(parent,lbl_to_nd[carrier_lbl]));
				float fs = f_score(carrier_lbl);
				if (fs > max_jest_score){
					max_jest_score = fs;
					jest_label = carrier_lbl;
				}
				if (!tree.is_leaf(lbl_to_nd[carrier_lbl])){			
					for (CT_NODE_T child : tree.get_children(lbl_to_nd[carrier_lbl])){
						string child_lbl = tree.get_label(child);
						Deque.push_back(child_lbl);
					}
				}
				Deque.pop_front();
			}
		}		
		consensus[c_idx] = jest_label;
		
		/*
		if (get_error){
			float edge_error = get_placement_error(gt_lbl_to_nd[jest_label], gt_lbl_to_nd[ground_truth], uni_tree);
			#pragma omp critical
			outFile << placement.n[0] << "\t" << jest_label << "\t" << edge_error <<"\n";
		}	
		else{
			#pragma omp critical
			outFile << placement.n[0] << "\t" << jest_label << "\t" << "N/A" <<"\n";
		}*/
	}
	//outFile.close();
	return consensus;
}

vector<placement_obj> placement_consensus (string jtree, string tree_string, vector<placement_obj>& placements, size_t num_threads, string dest_path, bool get_error, bool edge_error, vector<string> fields){

	cout << "Obtaining Placement Consensus!\n";
	vector<string> consensus = get_placement_consensus(jtree, tree_string, placements, num_threads, fields);
	cout << "Consensus Obtained!\nObtaining distal and pendant lengths!\n";
	size_t en_idx = distance(fields.begin(), find(fields.begin(), fields.end(), "edge_num"));
	size_t lwr_idx = distance(fields.begin(), find(fields.begin(), fields.end(), "like_weight_ratio"));
	size_t plen_idx = distance(fields.begin(), find(fields.begin(), fields.end(), "pendant_length"));
	size_t dlen_idx = distance(fields.begin(), find(fields.begin(), fields.end(), "distal_length"));

	vector<vector<float>> lengths = find_length(jtree, tree_string, placements, consensus, num_threads, lwr_idx, en_idx);
	//for(float d : distals) cout <<d<<", ";
	//cout << "\n";	
	vector<placement_obj> cplacements(placements.size(), placement_obj());
	for (size_t c_idx = 0; c_idx < consensus.size(); c_idx++){
		placement_obj pobj({placements[c_idx].n[0]}, vector<vector<float>>(1, vector<float>(fields.size(), 0.0)));
		int ixn = jtree.find(consensus[c_idx]);
		string jtree_substring = jtree.substr(ixn,jtree.length()-ixn);
		int lbracket = jtree_substring.find("{");
		int rbracket = jtree_substring.find("}");
		int con_edge = stoi(jtree_substring.substr(lbracket+1,rbracket-lbracket-1));
	
		// Note that we we give default 0.0 for likelihood, distal and pendant length	
		pobj.p[0][en_idx] = (float)con_edge;
		pobj.p[0][lwr_idx] = (float)1.0;
		pobj.p[0][plen_idx] = (ceil(lengths[1][c_idx] * 10000)/(10000));
		pobj.p[0][dlen_idx] = (ceil(lengths[0][c_idx] * 10000)/(10000));
		cplacements[c_idx] = pobj;
	}
	if (get_error)
		placement_error_multilabel(jtree, tree_string, placements, consensus, num_threads, edge_error, dest_path, fields);
	return cplacements;
}		

/*
 * get_sub_genomes returns 2 vectors:
 * 	0th index: vector of genomes
 * 	1st index: vector of consensus placement
 *
 */
vector<vector<string>> get_subgenome_consensus(string jtree, string tree_string, vector<placement_obj>& placements, size_t num_threads, vector<string> fields, string genome_map){
	
	size_t en_idx = distance(fields.begin(), find(fields.begin(), fields.end(), "edge_num"));
	size_t lwr_idx = distance(fields.begin(), find(fields.begin(), fields.end(), "like_weight_ratio"));
	size_t plen_idx = distance(fields.begin(), find(fields.begin(), fields.end(), "pendant_length"));
	size_t dlen_idx = distance(fields.begin(), find(fields.begin(), fields.end(), "distal_length"));
	size_t l_idx = distance(fields.begin(), find(fields.begin(), fields.end(), "likelihood"));
	
	compact_tree tree(tree_string.c_str(),false,true,true,0);
	map<string, CT_NODE_T> lbl_to_nd = label_to_node(tree);
	
	compact_tree uni_tree;
	map<string, CT_NODE_T> gt_lbl_to_nd;	
	regex gmap;
	if (genome_map != "")
		gmap = regex(genome_map);
	else
		gmap = regex(".+");

	// Redesigned to handle multiple genomes with multiple subgenomes
	map<string, map<string,float>> lbl_to_l_ratio;
	map<string, float> p_l_ratio_sum;

	size_t ol_threads = min(num_threads, placements.size());
	omp_set_num_threads(ol_threads);
	smatch gmatch;
	//#pragma omp parallel for reduction(+: p_l_ratio_sum)
	#pragma omp parallel
	for (size_t idx = 0; idx < placements.size(); idx++){
		placement_obj &placement = placements[idx];
		// TODO genome extraction via regex
		regex_match(placement.n[0], gmatch, gmap);
		for (size_t p_idx = 0; p_idx < placement.p.size(); p_idx++){
			int edge = placement.p[p_idx][en_idx];
			int ixe = jtree.find("{"+to_string(edge)+"}");
			string tree_substring = jtree.substr(0,ixe);
			int ixs = max({(int)tree_substring.rfind(','),
						(int)tree_substring.rfind('('),
						(int)tree_substring.rfind(')')});
			float likelihood = placement.p[p_idx][lwr_idx];
			vector<string> jtree_split = split_string(jtree.substr(ixs+1, ixe-ixs-1), ':');
			string lbl_placement = jtree_split[0];
			#pragma omp critical
			p_l_ratio_sum[gmatch.str(1)] += likelihood;
			lbl_to_l_ratio[gmatch.str(1)][lbl_placement] += likelihood;
		}
	}
	int c_idx = 0;
	vector<vector<string>> consensus(2, vector<string>(p_l_ratio_sum.size(),""));
	// Testing out multithreading through simple traversal
	#pragma omp parallel
	for (auto &pair : lbl_to_l_ratio){	
		if (p_l_ratio_sum[pair.first] == 0.0){
			cout << "Zero Placements found, unable to find a consensus!\n";
			//consensus[pair.first] = "N/A";
			consensus[0][c_idx] = pair.first;
			consensus[1][c_idx] = "N/A";
			c_idx+=1;
			continue;
		}
		// nice trick to normalize the likelihood weight ratio
		auto normalize_p_l_ratio = [&p_l_ratio_sum, &pair](auto &n){n = n/(p_l_ratio_sum[pair.first]+1e-10f);};
		vector<string> p_lbl(pair.second.size()); vector<float> p_l_ratio(pair.second.size());
		p_lbl.reserve(pair.second.size()); p_l_ratio.reserve(pair.second.size());
		for (auto &pair2: pair.second){
			p_lbl.push_back(pair2.first);
			p_l_ratio.push_back(pair2.second);
		}
		for_each(p_l_ratio.begin(), p_l_ratio.end(), normalize_p_l_ratio);
	
		// step 2: perform placement consensus
		string jest_label;
		float max_jest_score = 0.0;
		
		if (p_lbl.size() == 1){
			jest_label = p_lbl[0];
		}
		else{
			vector<CT_NODE_T> node_placements;
			for (size_t i =0; i < p_lbl.size(); i++)
				node_placements.push_back(lbl_to_nd[p_lbl[i]]);

			size_t ref_placement_idx = distance(p_l_ratio.begin(), max_element(p_l_ratio.begin(), p_l_ratio.end()));
			CT_NODE_T ref_placement = lbl_to_nd[p_lbl[ref_placement_idx]];
			float weighted_score = 0.0;
			for (size_t j = 0; j < node_placements.size(); j++){
				weighted_score += abs(tree.calc_dist(ref_placement, node_placements[j])) * p_l_ratio[j];
			}
			float w_gamma  = 1/(weighted_score + 1e-8f);
			map<string, float> TP_node, FP_node, FN_node;
			map<string, size_t> count;
			auto f_score = [&TP_node, &FP_node, &FN_node ](string lbl) -> float {
					if(TP_node[lbl] > 0.0) return 2*TP_node[lbl]/(2*TP_node[lbl] + FP_node[lbl] + FN_node[lbl]); 
					else return 0.0;
			};		
			deque<string> Deque;
			for (string p : p_lbl) Deque.push_back(p);
				string carrier_lbl;
			while (Deque.size() != 0){
				carrier_lbl = Deque.front();
				count[carrier_lbl] += 1;
				auto idx = find(p_lbl.begin(), p_lbl.end(), carrier_lbl);
				if (idx != p_lbl.end()){
					size_t p_idx = distance(p_lbl.begin(), idx);	
					TP_node[carrier_lbl] = p_l_ratio[p_idx]*w_gamma;
					FP_node[carrier_lbl] = (1.0-p_l_ratio[p_idx])*w_gamma;
				}else{
					TP_node[carrier_lbl] = 0.0;
					FP_node[carrier_lbl] = w_gamma;	
				}
				if (!tree.is_leaf(lbl_to_nd[carrier_lbl])){
					for (CT_NODE_T child : tree.get_children(lbl_to_nd[carrier_lbl])){
						string child_lbl = tree.get_label(child);
						TP_node[carrier_lbl] += TP_node[child_lbl]*exp(-w_gamma*tree.calc_dist(lbl_to_nd[carrier_lbl],child));
						FP_node[carrier_lbl] += FP_node[child_lbl]*exp(-w_gamma*tree.calc_dist(lbl_to_nd[carrier_lbl],child));
					}
				}
				if (count[carrier_lbl] == p_lbl.size())
					break;
				else if (tree.get_parent(lbl_to_nd[carrier_lbl]) != NULL_NODE)
					Deque.push_back(tree.get_label(tree.get_parent(lbl_to_nd[carrier_lbl])));
				Deque.pop_front();		
			}
			FN_node[carrier_lbl] = 0.0;
			jest_label = carrier_lbl;

			
			if (carrier_lbl == tree.get_label(tree.get_root()))
				max_jest_score = 0.0;
			else
				max_jest_score = f_score(carrier_lbl);
			Deque.clear();
			for (CT_NODE_T child : tree.get_children(lbl_to_nd[carrier_lbl])){
				string child_lbl = tree.get_label(child);
				Deque.push_back(child_lbl);
			}
			while (Deque.size() != 0){
				carrier_lbl = Deque.front();
				CT_NODE_T parent = tree.get_parent(lbl_to_nd[carrier_lbl]);
				string parent_lbl = tree.get_label(parent);
				FN_node[carrier_lbl] = (FN_node[parent_lbl] + FP_node[parent_lbl] -
					(FP_node[carrier_lbl] * exp(-w_gamma * tree.calc_dist(parent,lbl_to_nd[carrier_lbl])))) *
					exp(-w_gamma * tree.calc_dist(parent,lbl_to_nd[carrier_lbl]));
				float fs = f_score(carrier_lbl);
				if (fs > max_jest_score){
					max_jest_score = fs;
					jest_label = carrier_lbl;
				}
				if (!tree.is_leaf(lbl_to_nd[carrier_lbl])){			
					for (CT_NODE_T child : tree.get_children(lbl_to_nd[carrier_lbl])){
						string child_lbl = tree.get_label(child);
						Deque.push_back(child_lbl);
					}
				}
				Deque.pop_front();
			}
		}
		//consensus[pair.first] = jest_label;
		consensus[0][c_idx] = pair.first;
		consensus[1][c_idx] = jest_label;
		c_idx+=1;
	}
	return consensus;
}

vector<placement_obj> subgenome_consensus (string jtree, string tree_string, vector<placement_obj>& placements, size_t num_threads, string dest_path, bool get_error, bool edge_error, vector<string> fields, string genome_map){

	cout << "Obtaining Placement Consensus!\n";
	vector<vector<string>> consensus = get_subgenome_consensus(jtree, tree_string, placements, num_threads, fields, genome_map);
	cout << "Consensus Obtained!\nObtaining distal and pendant lengths!\n";
	size_t en_idx = distance(fields.begin(), find(fields.begin(), fields.end(), "edge_num"));
	size_t lwr_idx = distance(fields.begin(), find(fields.begin(), fields.end(), "like_weight_ratio"));
	size_t plen_idx = distance(fields.begin(), find(fields.begin(), fields.end(), "pendant_length"));
	size_t dlen_idx = distance(fields.begin(), find(fields.begin(), fields.end(), "distal_length"));

	//vector<vector<float>> lengths = find_length(jtree, tree_string, placements, consensus[1], num_threads, lwr_idx, en_idx);
	vector<placement_obj> cplacements(consensus[0].size(), placement_obj());
	cout << "Found " << consensus[0].size() << " different genomes!\n";
	
	for (size_t c_idx = 0; c_idx < consensus[0].size(); c_idx++){
		placement_obj pobj({consensus[0][c_idx]}, vector<vector<float>>(1, vector<float>(fields.size(), 0.0)));
		cout << consensus[1][c_idx]<< "\n";
		int ixn = jtree.find(consensus[1][c_idx]);
		string jtree_substring = jtree.substr(ixn,jtree.length()-ixn);
		int lbracket = jtree_substring.find("{");
		int rbracket = jtree_substring.find("}");
		int con_edge = stoi(jtree_substring.substr(lbracket+1,rbracket-lbracket-1));
		//vector<vector<float>> p(1, vector<float>(fields.size(), 0.0));
		pobj.p[0][en_idx] = (float)con_edge;
		pobj.p[0][lwr_idx] = (float)1.0;
		//p[0][plen_idx] = (float) (ceil(lengths[1][c_idx] * 10000)/(10000));
		//p[0][dlen_idx] = (float) (ceil(lengths[0][c_idx] * 10000)/(10000));
		//placement_obj pobj({consensus[0][c_idx]}, p);
		//cplacements.insert(c_idx, pobj);
		cplacements[c_idx] = pobj;
	}
	return cplacements;
}
/*
map<string, float> placement_error(vector<string> placements, string tree_string, string ground_truth, size_t num_threads, bool edge_error, string dest_path, bool is_path){

	ofstream outFile(dest_path, ios::out);
	compact_tree tree;

	if (is_path)
		tree = compact_tree(tree_string,true,true,true,0);
	else
		tree = compact_tree(tree_string,false,true,true,0);

	map<string, CT_NODE_T> lbl_to_node = label_to_node(tree);
	if (edge_error == true){
		for (CT_NODE_T node = 0; node < tree.get_num_nodes(); ++node) tree.set_edge_length(node,1.0);
		outFile << "placement\tground truth\tedge error\n";
	}
	else{
		outFile << "placement\tground truth\tbranch length error\n";
	}
	vector<pair<string,float>> errors(placements.size());
	size_t ol_threads = min(num_threads,placements.size());
	omp_set_nested(1);
	omp_set_num_threads(ol_threads);
	#pragma omp parallel for
	for (size_t idx = 0; idx < placements.size(); idx++){


}


map<string, float> placement_error(vector<string> placements,string tree_string, vector<string> ground_truth, size_t num_threads, bool edge_error, string dest_path, bool is_path){

}
*/
