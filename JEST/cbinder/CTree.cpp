#include "compact_tree.h"
#include "CTree_utils.h"
#include <string>
#include <map>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <tuple>
#include <cfloat>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cassert>
#include <omp.h>

using namespace std;
namespace fs = std::filesystem;


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
		//uncertainties.push_back(accumulate(ref_distances.begin(), ref_distances.end(),0.0)/random_placements);
		uncertainties[i] = accumulate(ref_distances.begin(), ref_distances.end(),0.0)/random_placements;
	}
	return uncertainties;
}
map<string,float> get_raw_uncertainty(string jtree, string tree_string, vector<placement_obj>& placements, 
								size_t num_threads, bool is_path){
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
			int edge = placement.p[p_idx][0];
			int ixe = jtree.find("{"+to_string(edge)+"}");
			string tree_substring = jtree.substr(0,ixe);
			int ixs = max({tree_substring.rfind(','),
						tree_substring.rfind('('),
						tree_substring.rfind(')')});
			float likelihood = placement.p[p_idx][4];
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
		//cout << p_lbl.size() << "\t";
		for (size_t i =0; i < p_lbl.size(); i++)
			node_placements.push_back(lbl_to_nd[p_lbl[i]]);

		// step 2: calculate raw uncertainty
		size_t ref_placement_idx = distance(p_l_ratio.begin(), max_element(p_l_ratio.begin(), p_l_ratio.end()));
		//cout << p_lbl[ref_placement_idx] <<"\t";
		CT_NODE_T ref_placement = lbl_to_nd[p_lbl[ref_placement_idx]];
		//cout << tree.get_label(ref_placement) << "\t";
		//cout << node_placements.size() << "\t";
		float weighted_score = 0.0;
		for (size_t j = 0; j < node_placements.size(); j++){
			//cout << node_placements[j] << "\t";
			//cout << tree.calc_dist(ref_placement, node_placements[j])  << "\t";
			weighted_score += abs(tree.calc_dist(ref_placement, node_placements[j])) * p_l_ratio[j];
		}
		uncertainty_scores[idx] = {placement.n[0], weighted_score};
	}
	map<string,float> placement_uncertainties(uncertainty_scores.begin(), uncertainty_scores.end());
	return placement_uncertainties;
}


map<string,float> get_uncertainty_pvalue(string jtree, string tree_string, vector<placement_obj>& placements, 
								float rp_mean, float rp_std, size_t num_threads, string dest_path, bool is_path){
	
	ofstream outFile(dest_path, ios::out);
	map<string,float> uncertainties = get_raw_uncertainty(jtree,tree_string,placements,num_threads,is_path);
	auto normalize = [&rp_mean, &rp_std](auto  kv) {kv.second = (kv.second-rp_mean)/(rp_std+1e-5f);};
	auto get_pvalue = [&rp_mean, &rp_std](float kv) {return 0.5+0.5*erf((kv-rp_mean)*M_SQRT1_2/(rp_std+1e-8f));};
	map<string,float> pval_uncertainties;
	outFile << "name\tuncertainty p-value\n";
	for(auto it = uncertainties.begin(); it != uncertainties.end(); ++it){
		pval_uncertainties[it->first] = get_pvalue(it->second);
		outFile << it->first << "\t" << pval_uncertainties[it->first] << "\n";
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

vector<string> placement_consensus(string jtree, string tree_string, vector<placement_obj>& placements, float gamma, size_t num_threads, string dest_path, bool get_error, string ground_truth, string gt_tree){
	
	ofstream outFile(dest_path, ios::out);
	
	vector<string> consensus(placements.size());
	consensus.reserve(placements.size());
	compact_tree tree(tree_string,false,true,true,0);

	map<string, CT_NODE_T> lbl_to_nd = label_to_node(tree);
	compact_tree uni_tree;
	map<string, CT_NODE_T> gt_lbl_to_nd;	
	if (get_error){
		if (gt_tree == ""){
			cout << "No other tree provided, Using tree from jplace file.\n";
			uni_tree = compact_tree(tree);
			gt_lbl_to_nd = lbl_to_nd;
		}
		else{
			uni_tree = compact_tree(gt_tree);
			gt_lbl_to_nd = label_to_node(uni_tree);
		}
		for (CT_NODE_T node = 0; node < uni_tree.get_num_nodes(); ++node) uni_tree.set_edge_length(node,1.0);
	}

	outFile << "name\tlabel\tedge error\n";
	
	omp_set_nested(1);
	//omp_set_num_threads(min(num_threads,placements.size()));
	size_t ol_threads = min(num_threads,placements.size());
	omp_set_num_threads(ol_threads);
	#pragma omp parallel for shared(tree,lbl_to_nd,uni_tree,gt_lbl_to_nd)
	for (size_t c_idx = 0; c_idx < placements.size(); c_idx++){
		placement_obj & placement = placements[c_idx];
		vector<string> p_lbl;
		vector<float> p_l_ratio;
		// step 1:  gather the node placements and their likelihood ratios
		//size_t il_threads = min(num_threads,placement.p.size());
		//omp_set_num_threads(il_threads)
		//#pragma omp parallel for
		for (size_t p_idx = 0; p_idx < placement.p.size(); p_idx++){
			int edge = (int)placement.p[p_idx][0];
			//cout<< "{"+to_string(edge)+"}" << "\t";
			int ixe = jtree.find("{"+to_string(edge)+"}");
			string tree_substring = jtree.substr(0,ixe);
			int ixs = max({tree_substring.rfind(','),
						tree_substring.rfind('('),
						tree_substring.rfind(')')});
			vector<string> jtree_split = split_string(jtree.substr(ixs+1, ixe-ixs-1), ':');
			p_lbl.push_back(jtree_split[0]);
			p_l_ratio.push_back(placement.p[p_idx][4]);
		}
		float p_l_ratio_sum = accumulate(p_l_ratio.begin(), p_l_ratio.end(), 0.0);
		if (p_l_ratio_sum == 0.0){
			outFile << placement.n[0] << "\t" << "N/A" << "N/A";
			continue;
		}
		// nice trick to normalize the likelihood weight ratio
		auto normalize_p_l_ratio = [& p_l_ratio_sum](float &n){n = n/p_l_ratio_sum;};
		for_each(p_l_ratio.begin(), p_l_ratio.end(), normalize_p_l_ratio);

		// step 2: perform placement consensus 	
		map<string, float> TP_node, TN_node, FN_node, FP_node;
		float max_f1_score = 0.0;
		string f1_label = tree.get_label(ROOT_NODE);
		string carrier_lbl;
		for (auto it = tree.postorder_begin(); it != tree.postorder_end(); ++it){
			string carrier_lbl = tree.get_label(*it);
			auto idx = find(p_lbl.begin(), p_lbl.end(), carrier_lbl);
			if(idx != p_lbl.end()){
				size_t p_idx = distance(p_lbl.begin(), idx);	
				TP_node[carrier_lbl] = p_l_ratio[p_idx]*gamma;
				FP_node[carrier_lbl] = (1.0-p_l_ratio[p_idx])*gamma;
			}else{
				TP_node[carrier_lbl] = 0.0;
				FP_node[carrier_lbl] = gamma;	
			}
			if (!tree.is_leaf(*it)){
				for (CT_NODE_T child: tree.get_children(*it)){
					string child_lbl = tree.get_label(child);
					TP_node[carrier_lbl] += TP_node[child_lbl]*exp(-gamma*tree.calc_dist(*it,child));
					FP_node[carrier_lbl] += FP_node[child_lbl]*exp(-gamma*tree.calc_dist(*it,child));
				}
			}
		}
		for(auto it = tree.levelorder_begin(); it != tree.levelorder_end(); ++it){
			carrier_lbl = tree.get_label(*it);
			if(tree.is_root(*it)){
				FN_node[carrier_lbl] = 0.0;
				TN_node[carrier_lbl] = gamma;
			}else{
				CT_NODE_T parent = tree.get_parent(*it);
				string parent_lbl = tree.get_label(parent);
				FN_node[carrier_lbl] = (FN_node[parent_lbl] + FP_node[parent_lbl] - (FP_node[carrier_lbl]*exp(-gamma*tree.calc_dist(*it,parent)))) * exp(-gamma*tree.calc_dist(*it,parent));
				TN_node[carrier_lbl] = (TN_node[parent_lbl] + TP_node[parent_lbl] - (TP_node[carrier_lbl]*exp(-gamma*tree.calc_dist(*it,parent)))) * exp(-gamma*tree.calc_dist(*it,parent));
			}
			float f1_score = (2* TP_node[carrier_lbl])/(2*TP_node[carrier_lbl]+FP_node[carrier_lbl]+FN_node[carrier_lbl]);
			if (f1_score > max_f1_score){
				max_f1_score = f1_score;
				f1_label = carrier_lbl;
			}
		}
		consensus[c_idx] = f1_label;
		if (get_error){
			float edge_error = get_placement_error(gt_lbl_to_nd[f1_label], gt_lbl_to_nd[ground_truth], uni_tree);
			#pragma omp critical
			outFile << placement.n[0] << "\t" << f1_label << "\t" << edge_error <<"\n";
		}	
		else{
			#pragma omp critical
			outFile << placement.n[0] << "\t" << f1_label << "\t" << "N/A" <<"\n";
		}
	}
	outFile.close();
	return consensus;
}

string gene_consensus(string jtree, string tree_string, vector<placement_obj>& placements, float gamma, size_t num_threads, bool get_error, string ground_truth, string gt_tree){
	
	compact_tree tree(tree_string.c_str(),false,true,true,0);
	map<string, CT_NODE_T> lbl_to_nd = label_to_node(tree);
	
	compact_tree uni_tree;
	map<string, CT_NODE_T> gt_lbl_to_nd;	
	if (get_error){
		if (gt_tree == ""){
			cout << "No other tree provided, Using tree from jplace file.\n";
			uni_tree = compact_tree(tree);
			gt_lbl_to_nd = lbl_to_nd;
		}
		else{
			uni_tree = compact_tree(gt_tree);
			gt_lbl_to_nd = label_to_node(uni_tree);
		}
		for (CT_NODE_T node = 0; node < uni_tree.get_num_nodes(); ++node) uni_tree.set_edge_length(node,1.0);
	}
	map<string,float> lbl_to_l_ratio;
	float p_l_ratio_sum = 0.0;

	size_t ol_threads = min(num_threads, placements.size());
	omp_set_num_threads(ol_threads);
	#pragma omp parallel for reduction(+: p_l_ratio_sum)
	for (size_t idx = 0; idx < placements.size(); idx++){
		placement_obj &placement = placements[idx];
		for (size_t p_idx = 0; p_idx < placement.p.size(); p_idx++){
			int edge = placement.p[p_idx][0];
			int ixe = jtree.find("{"+to_string(edge)+"}");
			string tree_substring = jtree.substr(0,ixe);
			int ixs = max({tree_substring.rfind(','),
						tree_substring.rfind('('),
						tree_substring.rfind(')')});
			float likelihood = placement.p[p_idx][4];
			vector<string> jtree_split = split_string(jtree.substr(ixs+1, ixe-ixs-1), ':');
			string lbl_placement = jtree_split[0];
			p_l_ratio_sum += likelihood;
			#pragma omp critical
			lbl_to_l_ratio[lbl_placement] += likelihood;
		}
	}	
	if (p_l_ratio_sum == 0.0){
		cout << "Zero Placements found, unable to find a consensus!\n";
		return "N/A";
	}
	// nice trick to normalize the likelihood weight ratio
	auto normalize_p_l_ratio = [& p_l_ratio_sum](auto &n){n = n/(p_l_ratio_sum+1e-10f);};
	vector<string> p_lbl(lbl_to_l_ratio.size()); vector<float> p_l_ratio(lbl_to_l_ratio.size());
	p_lbl.reserve(lbl_to_l_ratio.size()); p_l_ratio.reserve(lbl_to_l_ratio.size());
	for (auto &pair: lbl_to_l_ratio){
		p_lbl.push_back(pair.first);
		p_l_ratio.push_back(pair.second);
	}
	for_each(p_l_ratio.begin(), p_l_ratio.end(), normalize_p_l_ratio);
	map<string, float> TP_node, TN_node, FN_node, FP_node, F1_node;
	float max_f1_score = 0.0;
	string f1_label = tree.get_label(ROOT_NODE);
	vector<CT_NODE_T>carrier; 		
	string carrier_lbl;
	for (auto it = tree.postorder_begin(); it != tree.postorder_end(); ++it){
		string carrier_lbl = tree.get_label(*it);
		auto idx = find(p_lbl.begin(), p_lbl.end(), carrier_lbl);
		if(idx != p_lbl.end()){
			size_t p_idx = distance(p_lbl.begin(), idx);	
			TP_node[carrier_lbl] = p_l_ratio[p_idx]*gamma;
			FP_node[carrier_lbl] = (1.0-p_l_ratio[p_idx])*gamma;
		}else{
			TP_node[carrier_lbl] = 0.0;
			FP_node[carrier_lbl] = gamma;	
		}
		if (!tree.is_leaf(*it)){
			for (CT_NODE_T child: tree.get_children(*it)){
				string child_lbl = tree.get_label(child);
				TP_node[carrier_lbl] += TP_node[child_lbl]*exp(-gamma*tree.calc_dist(*it,child));
				FP_node[carrier_lbl] += FP_node[child_lbl]*exp(-gamma*tree.calc_dist(*it,child));
			}
		}
	}
	for(auto it = tree.levelorder_begin(); it != tree.levelorder_end(); ++it){
		carrier_lbl = tree.get_label(*it);
		if(tree.is_root(*it)){
			FN_node[carrier_lbl] = 0.0;
			TN_node[carrier_lbl] = gamma;
		}else{
			CT_NODE_T parent = tree.get_parent(*it);
			string parent_lbl = tree.get_label(parent);
			FN_node[carrier_lbl] = (FN_node[parent_lbl] + FP_node[parent_lbl] - (FP_node[carrier_lbl]*exp(-gamma*tree.calc_dist(*it,parent)))) * exp(-gamma*tree.calc_dist(*it,parent));
			TN_node[carrier_lbl] = (TN_node[parent_lbl] + TP_node[parent_lbl] - (TP_node[carrier_lbl]*exp(-gamma*tree.calc_dist(*it,parent)))) * exp(-gamma*tree.calc_dist(*it,parent));
		}
		float f1_score = (2* TP_node[carrier_lbl])/(2*TP_node[carrier_lbl]+FP_node[carrier_lbl]+FN_node[carrier_lbl]);
		F1_node[carrier_lbl] = f1_score;
		if (f1_score > max_f1_score){
			max_f1_score = f1_score;
			f1_label = carrier_lbl;
		}
	}
	if (get_error){
		float edge_error = get_placement_error(gt_lbl_to_nd[f1_label], gt_lbl_to_nd[ground_truth], uni_tree);
		cout << f1_label << "\t" << edge_error <<"\n";
	}
	return f1_label;
}
