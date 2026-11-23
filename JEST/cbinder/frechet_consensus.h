#ifndef JEST_FRECHET_CONSENSUS_H
#define JEST_FRECHET_CONSENSUS_H
#include <memory>
#include <string>
#include <vector>

struct FrechetPoint {
    unsigned long long edge_num;
    double distal_length;
};

struct FrechetResult {
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
    // Shared result container; these fields are used only by exponential_mode.
    double kernel_score = 0;
    size_t kernel_tie_count = 0;
    double kernel_tie_tolerance = 0;
};

// Immutable index, reusable across batches; no legacy float placement conversion.
class FrechetTreeIndex {
public:
    explicit FrechetTreeIndex(const std::string& newick);
    ~FrechetTreeIndex();
    std::string edge_label(unsigned long long edge_num) const;
    double edge_length(unsigned long long edge_num) const;
    // Uniform with respect to branch length, including any positive root stem.
    std::vector<FrechetPoint> random_points(unsigned long long count, unsigned long long seed) const;
    std::vector<double> mean_distance_batch(
        const std::vector<unsigned long long>& edges,
        const std::vector<double>& distal,
        const std::vector<double>& weights,
        const std::vector<unsigned long long>& offsets,
        const std::vector<unsigned long long>& reference_edges,
        const std::vector<double>& reference_distal,
        int threads) const;
    std::vector<FrechetResult> solve_batch(
        const std::vector<unsigned long long>& edges,
        const std::vector<double>& distal,
        const std::vector<double>& weights,
        const std::vector<unsigned long long>& offsets,
        const std::vector<unsigned long long>& query_counts,
        int threads, bool profile = false) const;
    std::vector<FrechetResult> exponential_mode_batch(
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
