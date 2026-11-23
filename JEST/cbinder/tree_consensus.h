#ifndef JEST_TREE_CONSENSUS_H
#define JEST_TREE_CONSENSUS_H
#include <memory>
#include <string>
#include <vector>

struct TreePoint {
    unsigned long long edge_num;
    double distal_length;
};

struct ConsensusResult {
    unsigned long long edge_num;
    double distal_length;
    double mean_squared_distance;
    double mean_distance;
    size_t query_count;
    size_t positive_row_count;
    // Optional per-group profiling; zero unless solve_batch(profile=true).
    double virtual_tree_seconds = 0;
    double moment_seconds = 0;
    double lookup_seconds = 0;
    double verification_seconds = 0;
    // Method-specific diagnostics, populated only by exponential_mode.
    double kernel_score = 0;
    size_t kernel_tie_count = 0;
    double kernel_tie_tolerance = 0;
};

// Immutable index, reusable across batches; no legacy float placement conversion.
class TreeConsensusIndex {
public:
    explicit TreeConsensusIndex(const std::string& newick);
    ~TreeConsensusIndex();
    std::string edge_label(unsigned long long edge_num) const;
    double edge_length(unsigned long long edge_num) const;
    // Uniform with respect to branch length, including any positive root stem.
    std::vector<TreePoint> random_points(unsigned long long count, unsigned long long seed) const;
    std::vector<double> mean_distance_batch(
        const std::vector<unsigned long long>& edges,
        const std::vector<double>& distal,
        const std::vector<double>& weights,
        const std::vector<unsigned long long>& offsets,
        const std::vector<unsigned long long>& reference_edges,
        const std::vector<double>& reference_distal,
        int threads) const;
    std::vector<ConsensusResult> solve_batch(
        const std::vector<unsigned long long>& edges,
        const std::vector<double>& distal,
        const std::vector<double>& weights,
        const std::vector<unsigned long long>& offsets,
        const std::vector<unsigned long long>& query_counts,
        int threads, bool profile = false) const;
    std::vector<ConsensusResult> exponential_mode_batch(
        const std::vector<unsigned long long>& edges,
        const std::vector<double>& distal,
        const std::vector<double>& weights,
        const std::vector<unsigned long long>& offsets,
        const std::vector<unsigned long long>& query_counts,
        double kernel_alpha, int threads, bool profile = false) const;
private:
    struct Impl;
    std::shared_ptr<const Impl> impl;
};
#endif
