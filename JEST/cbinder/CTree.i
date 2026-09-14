%module CTree
%{
    #define SWIG_FILE_WITH_INIT
    #include "compact_tree.h"
    #include "CTree_utils.h"
    #include "map"
    #include "vector"
    #include "string"
    using namespace std;
    extern vector<string> split_string(string s, char delim);
    extern map<string, CT_NODE_T> label_to_node(compact_tree tree); 
    extern map<string, vector<string>> label_children(string tree_string, bool is_path);
    extern vector<float> get_random_placements_uncertainty (string tree_string, size_t trials, size_t random_placements, size_t num_threads, bool is_path);
    extern float get_placement_error(CT_NODE_T u, CT_NODE_T v, compact_tree tree);
    extern map<string,float> get_raw_uncertainty (string jtree, string tree_string, vector<placement_obj>&, size_t num_threads, bool is_path);
    extern map<string,float> get_uncertainty_pvalue(string jtree, string tree_string, vector<placement_obj>&, float rp_mean, float rp_std , size_t num_threads, string dest_path, bool is_path);
    extern vector<string> placement_consensus (string jtree, string tree_string, vector<placement_obj>&, float gamma, size_t num_threads, string dest_path, bool get_error, string ground_truth, string gt_tree);
    extern string gene_consensus (string jtree, string tree_string, vector<placement_obj>&, float gamma, size_t num_threads, bool get_error, string ground_truth, string gt_tree);
%}
%include <std_string.i>
%include <std_map.i>
%include <std_vector.i>
%include <std_pair.i>
%include <stdint.i>
//%include <stdexcept>
//%include <exception.i>

%template(StrVector) std::vector<std::string>;
%template(FloatVector) std::vector<float>;
%template(DoubleVector) std::vector<double>;
%template(LabelChildrenMap) std::map<std::string, std::vector<std::string>>;
%template(Placement) std::vector<std::vector<float>>;
%template(PlacementUncertainties) std::map<std::string, float>;
%typemap(in) size_t{
    long long temp_val;
    if (!PyLong_Check($input)) {
        PyErr_SetString(PyExc_TypeError, "Expected an integer for size_t");
        SWIG_fail;
    }
    temp_val = PyLong_AsLongLong($input);
    if (temp_val < 0 || (unsigned long long) temp_val > SIZE_MAX){
        PyErr_SetString(PyExc_TypeError, "Out of Bounds, value either too large or is negative");
    }
    $1 = (size_t) temp_val;
}


%typemap (in) std::vector<placement_obj>& (std::vector<placement_obj> temp){
    Py_ssize_t nplacements = PyList_Size($input);
    temp.reserve(nplacements);

    for (Py_ssize_t k = 0; k < nplacements; ++k){
        PyObject *place = PyList_GetItem($input,k);
        
        if (!PyDict_Check(place)) {
            PyErr_SetString(PyExc_TypeError, "List elements must be dictionaries");
            SWIG_fail;
        }

        placement_obj p_obj;

        PyObject *nlist = PyDict_GetItemString(place, "n");
        if (nlist && PyList_Check(nlist)){
            Py_ssize_t nlen = PyList_Size(nlist);
            p_obj.n.reserve(nlen);
            for (Py_ssize_t i = 0; i < nlen; ++i){
                PyObject *item = PyList_GetItem(nlist,i);
                if (PyUnicode_Check(item))    
                    p_obj.n.push_back(PyUnicode_AsUTF8(item));
            }
        }
    
        PyObject *plist = PyDict_GetItemString(place, "p");
        if (plist && PyList_Check(plist)){
            Py_ssize_t plen = PyList_Size(plist);
            p_obj.p.resize(plen);
            for (Py_ssize_t i = 0; i < plen; ++i){
                PyObject *placement = PyList_GetItem(plist, i);
                Py_ssize_t pfields = PyList_Size(placement);
                p_obj.p[i].reserve(pfields);
                //std::vector<float> query_p;
                //query_p.reserve(pfields);
                for (Py_ssize_t j = 0; j < pfields; ++j) {
                    PyObject *py_val = PyList_GetItem(placement, j);
                    p_obj.p[i].push_back((float)PyFloat_AsDouble(py_val));
                }
                //p_obj.p.push_back(std::move(query_p)); 
            }
        }
        temp.push_back(p_obj);
    }
    $1 = &temp;
}


%include "CTree_utils.h"
%template(PlacementsList) std::vector<placement_obj>;

%newobject label_children;
%newobject get_random_placements;
%newobject get_raw_uncertainty;
%newobject get_uncertainty_pvalue;
%newobject placement_consensus;
%newobject gene_consensus;

std::vector<float> get_random_placements_uncertainty (std::string tree_string, size_t trials, size_t random_placements, size_t num_threads, bool is_path);
std::map<std::string, std::vector<std::string>> label_children(std::string tree_string, bool is_path);
std::map<std::string,float> get_raw_uncertainty (std::string jtree, std::string tree_string, std::vector<placement_obj>&, size_t num_threads, bool is_path);
std::map<std::string,float> get_uncertainty_pvalue(std::string jtree, std::string tree_string, std::vector<placement_obj>&, float rp_mean, float rp_std , size_t num_threads, std::string dest_path, bool is_path);
std::vector<std::string> placement_consensus (std::string jtree, std::string tree_string, vector<placement_obj>&, float gamma, size_t num_threads, std::string dest_path, bool get_error, std::string ground_truth, std::string gt_tree);
std::string gene_consensus (std::string jtree, std::string tree_string, vector<placement_obj>&, float gamma, size_t num_threads, bool get_error, std::string ground_truth, std::string gt_tree);

