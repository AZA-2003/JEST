#include "tree_consensus.h"
#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cerrno>
#include <chrono>
#include <random>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
using Real = long double;
using Id = unsigned long long;
struct Raw { int parent = -1; double length = 0; Id id = 0; bool annotated = false; std::string label; };
struct Parser {
    const std::string& s; size_t pos = 0;
    void space() { while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos; }
    char peek() { space(); return pos < s.size() ? s[pos] : '\0'; }
    void suffix(Raw& r) {
        space();
        if (peek() == '\'') {
            ++pos; bool closed = false;
            while (pos < s.size()) {
                if (s[pos++] == '\'') {
                    if (pos < s.size() && s[pos] == '\'') { r.label += '\''; ++pos; }
                    else { closed = true; break; }
                } else r.label += s[pos-1];
            }
            if (!closed) throw std::invalid_argument("Unclosed quoted Newick label");
        } else {
            size_t start = pos;
            while (pos < s.size() && std::string(":{},();[]").find(s[pos]) == std::string::npos) ++pos;
            size_t end = pos;
            while (end > start && std::isspace(static_cast<unsigned char>(s[end-1]))) --end;
            r.label = s.substr(start, end-start);
        }
        if (peek() == ':') {
            ++pos; space(); char* end = nullptr; errno = 0;
            r.length = std::strtod(s.c_str()+pos, &end);
            if(end == s.c_str()+pos || (errno == ERANGE && r.length == 0))
                throw std::invalid_argument("Invalid or underflowing Newick branch length");
            pos = static_cast<size_t>(end-s.c_str());
            if (!std::isfinite(r.length) || r.length < 0) throw std::invalid_argument("Tree lengths must be finite and nonnegative");
        }
        if (peek() == '{') {
            ++pos; size_t start = pos;
            while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) ++pos;
            if (start == pos || pos == s.size() || s[pos] != '}') throw std::invalid_argument("Invalid Newick edge annotation");
            try { r.id = std::stoull(s.substr(start, pos-start)); }
            catch (...) { throw std::invalid_argument("Edge ID exceeds uint64 range"); }
            r.annotated = true; ++pos;
        }
    }
    std::vector<Raw> parse() {
        std::vector<Raw> nodes; std::vector<int> stack;
        bool need_node = true, done = false;
        while (!done) {
            if (need_node) {
                char c = peek();
                if (!c || c == ')' || c == ',' || c == ';') throw std::invalid_argument("Expected Newick node");
                int v = static_cast<int>(nodes.size()); nodes.emplace_back();
                nodes[v].parent = stack.empty() ? -1 : stack.back();
                if (c == '(') { ++pos; stack.push_back(v); continue; }
                suffix(nodes[v]); need_node = false;
            }
            char c = peek();
            if (c == ',' && !stack.empty()) { ++pos; need_node = true; }
            else if (c == ')' && !stack.empty()) {
                ++pos; int v = stack.back(); stack.pop_back(); suffix(nodes[v]);
            } else if (c == ';' && stack.empty()) { ++pos; done = true; }
            else throw std::invalid_argument("Malformed annotated Newick tree");
        }
        if (peek()) throw std::invalid_argument("Trailing content after Newick tree");
        return nodes;
    }
};
struct Edge { int parent, child; double length; Id id; std::string label; };
struct Node {
    int parent = 0, topological_depth = 0, tin = 0, tout = 0, first = 0;
    Real root_distance = 0, root_distance_correction = 0;
    std::vector<int> children;
    Id canonical = std::numeric_limits<Id>::max();
    double canonical_distal = 0;
    bool has_canonical = false;
    int incoming = -1;
};
struct WeightedPoint { int edge_index; Real distal, weight; };
struct VirtualLocation { int anchor; Real root_distance; Real anchor_distal = 0; }; // vertex or point above anchor
struct VirtualNode {
    VirtualLocation loc; Real subtree_mass = 0, own_mass = 0, down_first_moment = 0, up_mass = 0, up_first_moment = 0, outside_first_moment = 0;
    std::vector<int> children;
    int original_edge_index = -1;
    Real original_distal = 0;
};
struct VirtualTree {
    std::vector<VirtualNode> nodes;
    std::vector<int> order;
};
// Neumaier accumulation preserves small nonnegative contributions.
struct CompensatedSum {
    Real value = 0, correction = 0;
    void add(Real x) {
        Real next = value+x;
        correction += std::abs(value)>=std::abs(x) ? (value-next)+x : (x-next)+value;
        value=next;
    }
    Real get() const { return value+correction; }
};
}

struct TreeConsensusIndex::Impl {
    std::vector<Node> nodes;
    std::vector<Edge> edges;
    std::unordered_map<Id, int> by_id;
    std::vector<Real> cumulative_lengths;
    std::vector<int> euler, logs;
    std::vector<std::vector<int>> rmq, ancestors;

    explicit Impl(const std::string& tree) {
        auto raw = Parser{tree}.parse();
        // A positive annotated root stem is an actual edge from a synthetic root.
        if (raw[0].length > 0 && !raw[0].annotated)
            throw std::invalid_argument("Positive root stem requires an edge annotation");
        nodes.emplace_back();
        std::vector<int> quotient(raw.size());
        for (size_t i = 0; i < raw.size(); ++i) {
            const auto& r = raw[i];
            if (i && !r.annotated) throw std::invalid_argument("Every non-root edge requires an edge ID");
            int p = r.parent < 0 ? 0 : quotient[r.parent];
            int v = p;
            if (r.length > 0) {
                v = static_cast<int>(nodes.size()); nodes.emplace_back();
                nodes[v].parent = p; nodes[v].topological_depth = nodes[p].topological_depth+1;
                nodes[v].root_distance = nodes[p].root_distance + static_cast<Real>(r.length);
                // Preserve the low-order part for constant-time local distances.
                // The original root-distance/coordinate validation remains unchanged.
                Real previous=nodes[p].root_distance, length=r.length, next=nodes[v].root_distance;
                Real error=previous>=length ? (previous-next)+length : (length-next)+previous;
                nodes[v].root_distance_correction=nodes[p].root_distance_correction+error;
                if (!std::isfinite(nodes[v].root_distance) || !(nodes[v].root_distance > nodes[p].root_distance)) throw std::invalid_argument("Tree length dynamic range exceeds index precision");
                nodes[p].children.push_back(v);
            }
            quotient[i] = v;
            if (r.annotated) {
                if (!by_id.emplace(r.id, static_cast<int>(edges.size())).second) throw std::invalid_argument("Duplicate tree edge ID");
                if (v != p) nodes[v].incoming = static_cast<int>(edges.size());
                edges.push_back({p, v, r.length, r.id, r.label});
                for (int endpoint : {p, v}) {
                    auto& n = nodes[endpoint];
                    if (!n.has_canonical || r.id < n.canonical) {
                        n.has_canonical = true; n.canonical = r.id;
                        n.canonical_distal = endpoint == v ? 0 : r.length;
                    }
                }
            }
        }
        if (edges.empty()) throw std::invalid_argument("Tree has no annotated edges");
        Real total_length = 0;
        for (const auto& edge : edges) {
            total_length += edge.length;
            if (!std::isfinite(total_length)) throw std::invalid_argument("Total tree length exceeds index precision");
            cumulative_lengths.push_back(total_length);
        }
        // Iterative Euler tour; topological depth, never weighted root distance, drives RMQ.
        std::vector<std::pair<int,size_t>> stack{{0,0}};
        int timer = 0; nodes[0].tin = timer++; nodes[0].first = 0; euler.push_back(0);
        while (!stack.empty()) {
            int u = stack.back().first;
            size_t& next = stack.back().second;
            if (next < nodes[u].children.size()) {
                int v = nodes[u].children[next++]; nodes[v].tin = timer++;
                nodes[v].first = static_cast<int>(euler.size()); euler.push_back(v); stack.push_back({v,0});
            } else {
                nodes[u].tout = timer; stack.pop_back();
                if (!stack.empty()) euler.push_back(stack.back().first);
            }
        }
        logs.resize(euler.size()+1);
        for (size_t i=2; i<logs.size(); ++i) logs[i]=logs[i/2]+1;
        rmq.push_back(euler);
        for (int j=1; (size_t(1)<<j)<=euler.size(); ++j) {
            size_t count=euler.size()-(size_t(1)<<j)+1; rmq.emplace_back(count);
            for (size_t i=0;i<count;++i) {
                int a=rmq[j-1][i], b=rmq[j-1][i+(size_t(1)<<(j-1))];
                rmq[j][i]=nodes[a].topological_depth<nodes[b].topological_depth?a:b;
            }
        }
        ancestors.emplace_back(nodes.size());
        for (size_t i=0;i<nodes.size();++i) ancestors[0][i]=nodes[i].parent;
        for (int j=1;(size_t(1)<<j)<nodes.size();++j) {
            ancestors.emplace_back(nodes.size());
            for(size_t i=0;i<nodes.size();++i) ancestors[j][i]=ancestors[j-1][ancestors[j-1][i]];
        }
    }
    bool ancestor(int u,int v) const { return nodes[u].tin<=nodes[v].tin && nodes[v].tin<nodes[u].tout; }
    int lca(int u,int v) const {
        int a=nodes[u].first,b=nodes[v].first; if(a>b)std::swap(a,b);
        int j=logs[b-a+1], x=rmq[j][a],y=rmq[j][b-(1<<j)+1];
        return nodes[x].topological_depth<nodes[y].topological_depth?x:y;
    }
    Real distance(int u,int v) const {
        int a=lca(u,v); CompensatedSum sum;
        sum.add(nodes[u].root_distance-nodes[a].root_distance); sum.add(nodes[u].root_distance_correction-nodes[a].root_distance_correction);
        sum.add(nodes[v].root_distance-nodes[a].root_distance); sum.add(nodes[v].root_distance_correction-nodes[a].root_distance_correction);
        return sum.get();
    }
    int edge_index(Id id) const {
        auto it=by_id.find(id); if(it==by_id.end())throw std::invalid_argument("Unknown placement edge ID: "+std::to_string(id)); return it->second;
    }
    WeightedPoint point(Id id, double distal, double weight = 1) const {
        int e=edge_index(id); double L=edges[e].length;
        if (!std::isfinite(distal) || !std::isfinite(weight) || weight<=0)
            throw std::invalid_argument("Batch weights must be positive and coordinates finite");
        double tol=32*std::numeric_limits<double>::epsilon()*L;
        if (distal < -tol || distal > L+tol)
            throw std::invalid_argument("Distal length outside tree edge");
        return {e,std::max(0.0,std::min(L,distal)),weight};
    }
    std::pair<int,Real> locate(VirtualLocation loc) const {
        int v=loc.anchor;
        if (loc.root_distance == nodes[v].root_distance) return {edge_index(nodes[v].canonical), nodes[v].canonical_distal};
        for(int j=static_cast<int>(ancestors.size())-1;j>=0;--j) {
            int a=ancestors[j][v]; if(nodes[a].root_distance>loc.root_distance)v=a;
        }
        int p=nodes[v].parent;
        if(nodes[p].root_distance==loc.root_distance) return {edge_index(nodes[p].canonical),nodes[p].canonical_distal};
        int e=nodes[v].incoming;
        if(e<0 || loc.root_distance<nodes[p].root_distance || loc.root_distance>nodes[v].root_distance)throw std::runtime_error("Invalid consensus coordinate");
        return {e,nodes[v].root_distance-loc.root_distance};
    }
    Real point_distance(int a,Real da,int b,Real db) const {
        if(a==b)return std::abs(da-db);
        const auto& x=edges[a]; const auto& y=edges[b];
        return std::min({da+distance(x.child,y.child)+db,
            da+distance(x.child,y.parent)+(y.length-db),
            (x.length-da)+distance(x.parent,y.child)+db,
            (x.length-da)+distance(x.parent,y.parent)+(y.length-db)});
    }
    VirtualTree build_virtual_tree(const std::vector<WeightedPoint>& points,size_t begin,size_t end,Real weight_scale=1) const {
        std::vector<int> vertices;
        std::map<int,std::map<Real,CompensatedSum>> interiors;
        std::map<int,CompensatedSum> vertex_mass;
        for(size_t i=begin;i<end;++i) {
            auto x=points[i]; x.weight/=weight_scale; auto e=edges[x.edge_index];
            vertices.push_back(e.parent); vertices.push_back(e.child);
            if(x.distal==0)vertex_mass[e.child].add(x.weight);
            else if(x.distal==e.length)vertex_mass[e.parent].add(x.weight);
            else interiors[e.child][x.distal].add(x.weight);
        }
        auto sort_unique=[&]() {
            std::sort(vertices.begin(),vertices.end(),[&](int a,int b){return nodes[a].tin<nodes[b].tin;});
            vertices.erase(std::unique(vertices.begin(),vertices.end()),vertices.end());
        };
        sort_unique(); size_t count=vertices.size();
        for(size_t i=1;i<count;++i)vertices.push_back(lca(vertices[i-1],vertices[i]));
        sort_unique();
        std::vector<VirtualNode> vt;
        std::vector<int> stack;
        for(int v:vertices) {
            int idx=static_cast<int>(vt.size()); vt.push_back({{v,nodes[v].root_distance}});
            vt[idx].original_edge_index=edge_index(nodes[v].canonical);
            vt[idx].original_distal=nodes[v].canonical_distal;
            auto m=vertex_mass.find(v); if(m!=vertex_mass.end())vt[idx].subtree_mass=m->second.get();
            while(!stack.empty() && !ancestor(vt[stack.back()].loc.anchor,v))stack.pop_back();
            if(!stack.empty()) {
                int parent=stack.back(); auto it=interiors.find(v);
                if(it!=interiors.end()) {
                    // Merge only identical original distal coordinates. Distinct
                    // inputs that collapse at root-distance precision are errors.
                    const auto& ps=it->second;
                    for(auto p=ps.rbegin();p!=ps.rend();++p) {
                        Real root_distance=nodes[v].root_distance-p->first;
                        if (!(root_distance>vt[parent].loc.root_distance && root_distance<nodes[v].root_distance))
                            throw std::invalid_argument("Placement coordinate dynamic range exceeds index precision");
                        int next=static_cast<int>(vt.size()); vt.push_back({{v,root_distance,p->first}});
                        vt[next].subtree_mass=p->second.get();
                        vt[next].original_edge_index=nodes[v].incoming; vt[next].original_distal=p->first;
                        vt[parent].children.push_back(next); parent=next;
                    }
                }
                vt[parent].children.push_back(idx);
            }
            stack.push_back(idx);
        }
        std::vector<int> order{0};
        for(size_t i=0;i<order.size();++i)for(int c:vt[order[i]].children)order.push_back(c);
        for(auto& v:vt)v.own_mass=v.subtree_mass;
        return {std::move(vt),std::move(order)};
    }
    ConsensusResult solve_frechet(const std::vector<WeightedPoint>& points,size_t begin,size_t end,Id query_count,bool profile) const {
        using Clock = std::chrono::steady_clock;
        auto started = profile ? Clock::now() : Clock::time_point{};
        if (end-begin == 1) {
            const auto& point = points[begin];
            const auto& edge = edges[point.edge_index];
            if (point.distal == 0 || point.distal == edge.length) {
                int vertex = point.distal == 0 ? edge.child : edge.parent;
                const auto& node = nodes[vertex];
                return {node.canonical, node.canonical_distal, 0, 0,
                        static_cast<size_t>(query_count), 1};
            }
            return {edge.id, static_cast<double>(point.distal), 0, 0,
                    static_cast<size_t>(query_count), 1};
        }
        auto tree=build_virtual_tree(points,begin,end);
        auto& vt=tree.nodes; auto& order=tree.order;
        auto built = profile ? Clock::now() : Clock::time_point{};
        for(auto it=order.rbegin();it!=order.rend();++it) {
            int u=*it;
            for(int v:vt[u].children) {
                Real length=vt[v].loc.root_distance-vt[u].loc.root_distance;
                vt[u].subtree_mass+=vt[v].subtree_mass; vt[u].down_first_moment+=vt[v].down_first_moment+length*vt[v].subtree_mass;
            }
        }
        Real W=vt[0].subtree_mass;
        if (!(W > 0) || !std::isfinite(W))throw std::invalid_argument("Total batch weight exceeds accumulator range");
        // Nonnegative prefix/suffix messages avoid total-minus-child cancellation.
        for(int u:order) {
            auto& cs=vt[u].children; size_t n=cs.size();
            std::vector<Real> sm(n+1),sh(n+1);
            for(size_t i=n;i>0;--i) {
                int v=cs[i-1]; Real len=vt[v].loc.root_distance-vt[u].loc.root_distance;
                sm[i-1]=sm[i]+vt[v].subtree_mass; sh[i-1]=sh[i]+vt[v].down_first_moment+len*vt[v].subtree_mass;
            }
            Real pm=vt[u].up_mass+vt[u].own_mass, ph=vt[u].up_first_moment;
            for(size_t i=0;i<n;++i) {
                int v=cs[i]; Real len=vt[v].loc.root_distance-vt[u].loc.root_distance;
                Real C=pm+sm[i+1], E=ph+sh[i+1];
                vt[v].up_mass=C; vt[v].up_first_moment=E+len*C; vt[v].outside_first_moment=E;
                pm+=vt[v].subtree_mass; ph+=vt[v].down_first_moment+len*vt[v].subtree_mass;
            }
        }
        // Strict convexity permits descent through the only negative direction.
        // If t spans the whole edge, the parent direction at the next vertex
        // cannot improve the objective; otherwise t is its unique interior zero.
        VirtualLocation winner=vt[0].loc; int u=0;
        while(true) {
            bool moved=false;
            for(int v:vt[u].children) {
                Real len=vt[v].loc.root_distance-vt[u].loc.root_distance;
                Real B=vt[v].down_first_moment+len*vt[v].subtree_mass;
                Real t=(B-vt[v].outside_first_moment)/W;
                if(t>0) {
                    if(t<len) {
                        winner={vt[v].loc.anchor,vt[u].loc.root_distance+t};
                        moved=false;
                    } else {
                        u=v; winner=vt[v].loc; moved=true;
                    }
                    break;
                }
            }
            if(!moved)break;
        }
        auto optimized = profile ? Clock::now() : Clock::time_point{};
        auto out=locate(winner);
        // Verify the serialized double coordinate, including endpoint rounding.
        out.second=static_cast<double>(out.second);
        const auto& edge=edges[out.first];
        if(out.second<0 || out.second>edge.length)
            throw std::runtime_error("Consensus coordinate exceeds original edge at index precision");
        if(out.second==0 || out.second==edge.length) {
            int v=out.second==0?edge.child:edge.parent;
            out={edge_index(nodes[v].canonical),nodes[v].canonical_distal};
        }
        auto located = profile ? Clock::now() : Clock::time_point{};
        Real mse=0,mean=0;
        for(size_t i=begin;i<end;++i) {
            Real d=point_distance(out.first,out.second,points[i].edge_index,points[i].distal);
            mean+=points[i].weight*d; mse+=points[i].weight*d*d;
        }
        if(!std::isfinite(mse/W) || mse/W>std::numeric_limits<double>::max())throw std::runtime_error("Consensus objective exceeds double range");
        ConsensusResult result{edges[out.first].id,static_cast<double>(out.second),static_cast<double>(mse/W),static_cast<double>(mean/W),static_cast<size_t>(query_count),end-begin};
        if (profile) {
            result.virtual_tree_seconds=std::chrono::duration<double>(built-started).count();
            result.moment_seconds=std::chrono::duration<double>(optimized-built).count();
            result.lookup_seconds=std::chrono::duration<double>(located-optimized).count();
            result.verification_seconds=std::chrono::duration<double>(Clock::now()-located).count();
        }
        return result;
    }
    ConsensusResult solve_exponential_mode(const std::vector<WeightedPoint>& points,size_t begin,size_t end,
                                   Id query_count,Real kernel_alpha,bool profile) const {
        using Clock=std::chrono::steady_clock;
        auto started=profile?Clock::now():Clock::time_point{};
        if(end-begin==1) {
            auto result=solve_frechet(points,begin,end,query_count,profile);
            result.kernel_score=1; result.kernel_tie_count=1;
            result.kernel_tie_tolerance=64*std::numeric_limits<Real>::epsilon();
            return result;
        }
        Real scale=0;
        for(size_t i=begin;i<end;++i)scale=std::max(scale,points[i].weight);
        auto tree=build_virtual_tree(points,begin,end,scale);
        auto& vt=tree.nodes; auto& order=tree.order;
        auto built=profile?Clock::now():Clock::time_point{};
        size_t count=vt.size(), support_count=0; int support=-1;
        CompensatedSum total;
        for(size_t i=0;i<count;++i) {
            total.add(vt[i].own_mass);
            if(vt[i].own_mass>0){++support_count; support=static_cast<int>(i);}
        }
        Real W=total.get();
        std::vector<Real> down_kernel_sum(count),up_kernel_sum(count),attenuation(count),candidate_scores(count);
        for(auto it=order.rbegin();it!=order.rend();++it) {
            int u=*it; CompensatedSum sum; sum.add(vt[u].own_mass);
            for(int v:vt[u].children) {
                CompensatedSum length_sum;
                length_sum.add(distance(vt[u].loc.anchor,vt[v].loc.anchor));
                length_sum.add(vt[u].loc.anchor_distal);length_sum.add(-vt[v].loc.anchor_distal);
                Real length=length_sum.get();
                if(length<0)throw std::runtime_error("Negative virtual segment length");
                attenuation[v]=std::exp(-kernel_alpha*length);
                sum.add(attenuation[v]*down_kernel_sum[v]);
            }
            down_kernel_sum[u]=sum.get();
        }
        Real maximum=0;
        for(int u:order) {
            const auto& cs=vt[u].children; size_t n=cs.size();
            std::vector<Real> suffix(n+1); CompensatedSum tail;
            for(size_t i=n;i>0;--i){tail.add(attenuation[cs[i-1]]*down_kernel_sum[cs[i-1]]);suffix[i-1]=tail.get();}
            CompensatedSum score; score.add(down_kernel_sum[u]);score.add(up_kernel_sum[u]);candidate_scores[u]=score.get();
            maximum=std::max(maximum,candidate_scores[u]);
            CompensatedSum prefix; prefix.add(up_kernel_sum[u]);prefix.add(vt[u].own_mass);
            for(size_t i=0;i<n;++i) {
                int v=cs[i]; CompensatedSum outside;outside.add(prefix.get());outside.add(suffix[i+1]);
                up_kernel_sum[v]=attenuation[v]*outside.get();
                prefix.add(attenuation[v]*down_kernel_sum[v]);
            }
        }
        // Compare every candidate with the numerical maximum, never a chain of ties.
        Real tolerance=64*std::numeric_limits<Real>::epsilon();
        int winner=-1; size_t ties=0;
        auto key=[&](int v){return std::make_pair(edges[vt[v].original_edge_index].id,vt[v].original_distal);};
        if(support_count==1){winner=support;ties=1;}
        else for(size_t i=0;i<count;++i)if(maximum-candidate_scores[i]<=tolerance*W) {
            ++ties;
            if(winner<0 || key(static_cast<int>(i))<key(winner))winner=static_cast<int>(i);
        }
        if(winner<0 || !(maximum>0))throw std::runtime_error("Invalid exponential mode score");
        auto optimized=profile?Clock::now():Clock::time_point{};
        // All candidates already have canonical vertex or original observed coordinates.
        int edge=vt[winner].original_edge_index; Real distal=vt[winner].original_distal;
        auto located=profile?Clock::now():Clock::time_point{};
        CompensatedSum kernel,mean,mse,weight;
        bool mean_overflow=false,mse_overflow=false;
        for(size_t i=begin;i<end;++i) {
            Real w=(points[i].weight/scale)/W;
            Real d=point_distance(edge,distal,points[i].edge_index,points[i].distal);
            weight.add(w); kernel.add(w*std::exp(-kernel_alpha*d));
            if(std::isfinite(w*d))mean.add(w*d);else mean_overflow=true;
            if(std::isfinite(w*d*d))mse.add(w*d*d);else mse_overflow=true;
        }
        Real normalized=kernel.get()/weight.get();
        Real check_tolerance=std::max(Real(1e-12),256*std::numeric_limits<Real>::epsilon()*count);
        if(!std::isfinite(normalized) || normalized<=0 || normalized>1+check_tolerance ||
           std::abs(normalized-candidate_scores[winner]/W)>check_tolerance)
            throw std::runtime_error("Exponential mode direct-score consistency check failed");
        // Optional comparison diagnostics must not invalidate a bounded kernel score.
        auto diagnostic=[](Real value,bool overflow) {
            return overflow || !std::isfinite(value) || value>std::numeric_limits<double>::max()
                ? std::numeric_limits<double>::infinity() : static_cast<double>(value);
        };
        ConsensusResult result{edges[edge].id,static_cast<double>(distal),
            diagnostic(mse.get()/weight.get(),mse_overflow),diagnostic(mean.get()/weight.get(),mean_overflow),
            static_cast<size_t>(query_count),end-begin};
        result.kernel_score=static_cast<double>(std::min(Real(1),normalized));
        result.kernel_tie_count=ties;result.kernel_tie_tolerance=static_cast<double>(tolerance);
        if(profile) {
            result.virtual_tree_seconds=std::chrono::duration<double>(built-started).count();
            result.moment_seconds=std::chrono::duration<double>(optimized-built).count();
            result.lookup_seconds=std::chrono::duration<double>(located-optimized).count();
            result.verification_seconds=std::chrono::duration<double>(Clock::now()-located).count();
        }
        return result;
    }

    std::vector<ConsensusResult> solve_batch_impl(const std::vector<Id>& ids,const std::vector<double>& distal,
        const std::vector<double>& weights,const std::vector<Id>& offsets,const std::vector<Id>& query_counts,
        int threads,bool profile,Real kernel_alpha=0) const {
        if(threads<=0)throw std::invalid_argument("num_threads must be positive");
        if(ids.size()!=distal.size() || ids.size()!=weights.size() || offsets.size()!=query_counts.size()+1 || offsets.empty() || offsets.front()!=0 || offsets.back()!=ids.size())throw std::invalid_argument("Invalid batch buffers or offsets");
        std::vector<WeightedPoint> points; points.reserve(ids.size());
        for(size_t i=0;i<ids.size();++i) {
            points.push_back(point(ids[i],distal[i],weights[i]));
        }
        for(size_t i=0;i<query_counts.size();++i)if(!query_counts[i] || offsets[i]>=offsets[i+1])throw std::invalid_argument("Empty consensus group");
        std::vector<ConsensusResult> results(query_counts.size()); std::vector<std::string> errors(query_counts.size());
        if(query_counts.empty())return results;
        int workers=static_cast<int>(std::min<size_t>(threads,query_counts.size()));
        #pragma omp parallel for num_threads(workers) schedule(dynamic)
        for(size_t i=0;i<query_counts.size();++i) {
            try { results[i]=(kernel_alpha>0 ? solve_exponential_mode(points,offsets[i],offsets[i+1],query_counts[i],kernel_alpha,profile) : solve_frechet(points,offsets[i],offsets[i+1],query_counts[i],profile)); }
            catch(const std::exception& e) { errors[i]=e.what(); }
            catch(...) { errors[i]="Unknown solver error"; }
        }
        for(size_t i=0;i<errors.size();++i)if(!errors[i].empty())throw std::runtime_error("Consensus group "+std::to_string(i)+": "+errors[i]);
        return results;
    }

};

TreeConsensusIndex::TreeConsensusIndex(const std::string& tree):impl(std::make_shared<Impl>(tree)){}
TreeConsensusIndex::~TreeConsensusIndex()=default;
std::string TreeConsensusIndex::edge_label(Id id) const {
    const auto& edge = impl->edges[impl->edge_index(id)];
    return edge.label.empty() ? "edge_"+std::to_string(id) : edge.label;
}
double TreeConsensusIndex::edge_length(Id id) const { return impl->edges[impl->edge_index(id)].length; }
std::vector<ConsensusResult> TreeConsensusIndex::solve_batch(const std::vector<Id>& ids,const std::vector<double>& distal,const std::vector<double>& weights,const std::vector<Id>& offsets,const std::vector<Id>& query_counts,int threads,bool profile) const {
    return impl->solve_batch_impl(ids,distal,weights,offsets,query_counts,threads,profile);
}

std::vector<ConsensusResult> TreeConsensusIndex::exponential_mode_batch(
    const std::vector<Id>& ids,const std::vector<double>& distal,const std::vector<double>& weights,
    const std::vector<Id>& offsets,const std::vector<Id>& query_counts,double kernel_alpha,int threads,bool profile) const {
    if(!std::isfinite(kernel_alpha) || kernel_alpha<=0)
        throw std::invalid_argument("kernel_alpha must be finite and positive");
    return impl->solve_batch_impl(ids,distal,weights,offsets,query_counts,threads,profile,kernel_alpha);
}


std::vector<TreePoint> TreeConsensusIndex::random_points(Id count, Id seed) const {
    std::mt19937_64 engine(seed);
    std::uniform_real_distribution<double> unit(0,1);
    std::vector<TreePoint> result; result.reserve(count);
    Real total=impl->cumulative_lengths.back();
    for(size_t i=0;i<count;++i) {
        if(total==0) {
            result.push_back({impl->nodes[0].canonical,impl->nodes[0].canonical_distal});
            continue;
        }
        Real target=unit(engine)*total;
        // Roundoff cannot turn an endpoint sample into an invalid edge index.
        if(target>=total)target=std::nextafter(total,Real(0));
        size_t e=std::upper_bound(impl->cumulative_lengths.begin(),impl->cumulative_lengths.end(),target)-impl->cumulative_lengths.begin();
        const auto& edge=impl->edges[e];
        result.push_back({edge.id,unit(engine)*edge.length});
    }
    return result;
}

std::vector<double> TreeConsensusIndex::mean_distance_batch(
    const std::vector<Id>& ids,const std::vector<double>& distal,
    const std::vector<double>& weights,const std::vector<Id>& offsets,
    const std::vector<Id>& reference_edges,const std::vector<double>& reference_distal,
    int threads) const {
    size_t groups=reference_edges.size();
    if(threads<=0)throw std::invalid_argument("num_threads must be positive");
    if(ids.size()!=distal.size() || ids.size()!=weights.size() || reference_distal.size()!=groups ||
       offsets.size()!=groups+1 || offsets.empty() || offsets.front()!=0 || offsets.back()!=ids.size())
        throw std::invalid_argument("Invalid distance batch buffers or offsets");
    for(size_t i=0;i<groups;++i)if(offsets[i]>=offsets[i+1])throw std::invalid_argument("Empty distance group");
    std::vector<WeightedPoint> points,refs;
    points.reserve(ids.size()); refs.reserve(groups);
    for(size_t i=0;i<ids.size();++i)points.push_back(impl->point(ids[i],distal[i],weights[i]));
    for(size_t i=0;i<groups;++i)refs.push_back(impl->point(reference_edges[i],reference_distal[i]));
    std::vector<double> result(groups);
    std::vector<std::string> errors(groups);
    if(!groups)return result;
    int workers=static_cast<int>(std::min<size_t>(threads,groups));
    #pragma omp parallel for num_threads(workers) schedule(dynamic)
    for(size_t i=0;i<groups;++i) {
        try {
            Real total=0,first=0;
            for(size_t j=offsets[i];j<offsets[i+1];++j) {
                first+=points[j].weight*impl->point_distance(refs[i].edge_index,refs[i].distal,points[j].edge_index,points[j].distal);
                total+=points[j].weight;
            }
            Real value=first/total;
            if(!std::isfinite(total) || !std::isfinite(value) || value>std::numeric_limits<double>::max())
                throw std::runtime_error("Mean distance exceeds accumulator range");
            result[i]=static_cast<double>(value);
        } catch(const std::exception& e) { errors[i]=e.what(); }
        catch(...) { errors[i]="Unknown distance evaluation error"; }
    }
    for(size_t i=0;i<groups;++i)if(!errors[i].empty())throw std::runtime_error(errors[i]);
    return result;
}
