#include "frechet_consensus.h"
#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cerrno>
#include <chrono>
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
struct Raw { int parent = -1; double length = 0; Id id = 0; bool annotated = false; };
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
                    if (pos < s.size() && s[pos] == '\'') ++pos;
                    else { closed = true; break; }
                }
            }
            if (!closed) throw std::invalid_argument("Unclosed quoted Newick label");
        } else {
            while (pos < s.size() && std::string(":{},();[]").find(s[pos]) == std::string::npos) ++pos;
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
struct Edge { int parent, child; double length; Id id; };
struct Node {
    int parent = 0, level = 0, tin = 0, tout = 0, first = 0;
    Real depth = 0;
    std::vector<int> children;
    Id canonical = std::numeric_limits<Id>::max();
    double canonical_distal = 0;
    bool has_canonical = false;
    int incoming = -1;
};
struct Point { int edge; Real distal, weight; };
struct Location { int anchor; Real depth; }; // vertex or point above anchor
struct VirtualNode {
    Location loc; Real mass = 0, own = 0, down = 0, up_mass = 0, up = 0, outside = 0;
    std::vector<int> children;
};
}

struct FrechetTreeIndex::Impl {
    std::vector<Node> nodes;
    std::vector<Edge> edges;
    std::unordered_map<Id, int> by_id;
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
                nodes[v].parent = p; nodes[v].level = nodes[p].level+1;
                nodes[v].depth = nodes[p].depth + static_cast<Real>(r.length);
                if (!std::isfinite(nodes[v].depth) || !(nodes[v].depth > nodes[p].depth)) throw std::invalid_argument("Tree length dynamic range exceeds index precision");
                nodes[p].children.push_back(v);
            }
            quotient[i] = v;
            if (r.annotated) {
                if (!by_id.emplace(r.id, static_cast<int>(edges.size())).second) throw std::invalid_argument("Duplicate tree edge ID");
                if (v != p) nodes[v].incoming = static_cast<int>(edges.size());
                edges.push_back({p, v, r.length, r.id});
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
        // Iterative Euler tour; topological depth, never weighted depth, drives RMQ.
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
                rmq[j][i]=nodes[a].level<nodes[b].level?a:b;
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
        return nodes[x].level<nodes[y].level?x:y;
    }
    Real distance(int u,int v) const {
        int a=lca(u,v); return (nodes[u].depth-nodes[a].depth)+(nodes[v].depth-nodes[a].depth);
    }
    int edge_index(Id id) const {
        auto it=by_id.find(id); if(it==by_id.end())throw std::invalid_argument("Unknown placement edge ID: "+std::to_string(id)); return it->second;
    }
    std::pair<int,Real> locate(Location loc) const {
        int v=loc.anchor;
        if (loc.depth == nodes[v].depth) return {edge_index(nodes[v].canonical), nodes[v].canonical_distal};
        for(int j=static_cast<int>(ancestors.size())-1;j>=0;--j) {
            int a=ancestors[j][v]; if(nodes[a].depth>loc.depth)v=a;
        }
        int p=nodes[v].parent;
        if(nodes[p].depth==loc.depth) return {edge_index(nodes[p].canonical),nodes[p].canonical_distal};
        int e=nodes[v].incoming;
        if(e<0 || loc.depth<nodes[p].depth || loc.depth>nodes[v].depth)throw std::runtime_error("Invalid consensus coordinate");
        return {e,nodes[v].depth-loc.depth};
    }
    Real point_distance(int a,Real da,int b,Real db) const {
        if(a==b)return std::abs(da-db);
        const auto& x=edges[a]; const auto& y=edges[b];
        return std::min({da+distance(x.child,y.child)+db,
            da+distance(x.child,y.parent)+(y.length-db),
            (x.length-da)+distance(x.parent,y.child)+db,
            (x.length-da)+distance(x.parent,y.parent)+(y.length-db)});
    }
    FrechetResult solve(const std::vector<Point>& points,size_t begin,size_t end,Id queries,bool profile) const {
        using Clock = std::chrono::steady_clock;
        auto started = profile ? Clock::now() : Clock::time_point{};
        if (end-begin == 1) {
            const auto& point = points[begin];
            const auto& edge = edges[point.edge];
            if (point.distal == 0 || point.distal == edge.length) {
                int vertex = point.distal == 0 ? edge.child : edge.parent;
                const auto& node = nodes[vertex];
                return {node.canonical, node.canonical_distal, 0, 0,
                        static_cast<size_t>(queries), 1};
            }
            return {edge.id, static_cast<double>(point.distal), 0, 0,
                    static_cast<size_t>(queries), 1};
        }
        std::vector<int> vertices;
        std::map<int,std::map<Real,Real>> interiors;
        std::map<int,Real> vertex_mass;
        for(size_t i=begin;i<end;++i) {
            auto x=points[i]; auto e=edges[x.edge];
            vertices.push_back(e.parent); vertices.push_back(e.child);
            if(x.distal==0)vertex_mass[e.child]+=x.weight;
            else if(x.distal==e.length)vertex_mass[e.parent]+=x.weight;
            else interiors[e.child][x.distal]+=x.weight;
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
            int idx=static_cast<int>(vt.size()); vt.push_back({{v,nodes[v].depth}});
            auto m=vertex_mass.find(v); if(m!=vertex_mass.end())vt[idx].mass=m->second;
            while(!stack.empty() && !ancestor(vt[stack.back()].loc.anchor,v))stack.pop_back();
            if(!stack.empty()) {
                int parent=stack.back(); auto it=interiors.find(v);
                if(it!=interiors.end()) {
                    // Merge only identical original distal coordinates. Distinct
                    // inputs that collapse at root-depth precision are errors.
                    const auto& ps=it->second;
                    for(auto p=ps.rbegin();p!=ps.rend();++p) {
                        Real depth=nodes[v].depth-p->first;
                        if (!(depth>vt[parent].loc.depth && depth<nodes[v].depth))
                            throw std::invalid_argument("Placement coordinate dynamic range exceeds index precision");
                        int next=static_cast<int>(vt.size()); vt.push_back({{v,depth}});
                        vt[next].mass=p->second;
                        vt[parent].children.push_back(next); parent=next;
                    }
                }
                vt[parent].children.push_back(idx);
            }
            stack.push_back(idx);
        }
        std::vector<int> order{0};
        for(size_t i=0;i<order.size();++i)for(int c:vt[order[i]].children)order.push_back(c);
        auto built = profile ? Clock::now() : Clock::time_point{};
        for(auto& v:vt)v.own=v.mass;
        for(auto it=order.rbegin();it!=order.rend();++it) {
            int u=*it;
            for(int v:vt[u].children) {
                Real length=vt[v].loc.depth-vt[u].loc.depth;
                vt[u].mass+=vt[v].mass; vt[u].down+=vt[v].down+length*vt[v].mass;
            }
        }
        Real W=vt[0].mass;
        if (!(W > 0) || !std::isfinite(W))throw std::invalid_argument("Total batch weight exceeds accumulator range");
        // Nonnegative prefix/suffix messages avoid total-minus-child cancellation.
        for(int u:order) {
            auto& cs=vt[u].children; size_t n=cs.size();
            std::vector<Real> sm(n+1),sh(n+1);
            for(size_t i=n;i>0;--i) {
                int v=cs[i-1]; Real len=vt[v].loc.depth-vt[u].loc.depth;
                sm[i-1]=sm[i]+vt[v].mass; sh[i-1]=sh[i]+vt[v].down+len*vt[v].mass;
            }
            Real pm=vt[u].up_mass+vt[u].own, ph=vt[u].up;
            for(size_t i=0;i<n;++i) {
                int v=cs[i]; Real len=vt[v].loc.depth-vt[u].loc.depth;
                Real C=pm+sm[i+1], E=ph+sh[i+1];
                vt[v].up_mass=C; vt[v].up=E+len*C; vt[v].outside=E;
                pm+=vt[v].mass; ph+=vt[v].down+len*vt[v].mass;
            }
        }
        // Strict convexity permits descent through the only negative direction.
        // If t spans the whole edge, the parent direction at the next vertex
        // cannot improve the objective; otherwise t is its unique interior zero.
        Location winner=vt[0].loc; int u=0;
        while(true) {
            bool moved=false;
            for(int v:vt[u].children) {
                Real len=vt[v].loc.depth-vt[u].loc.depth;
                Real B=vt[v].down+len*vt[v].mass;
                Real t=(B-vt[v].outside)/W;
                if(t>0) {
                    if(t<len) {
                        winner={vt[v].loc.anchor,vt[u].loc.depth+t};
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
            Real d=point_distance(out.first,out.second,points[i].edge,points[i].distal);
            mean+=points[i].weight*d; mse+=points[i].weight*d*d;
        }
        if(!std::isfinite(mse/W) || mse/W>std::numeric_limits<double>::max())throw std::runtime_error("Consensus objective exceeds double range");
        FrechetResult result{edges[out.first].id,static_cast<double>(out.second),static_cast<double>(mse/W),static_cast<double>(mean/W),static_cast<size_t>(queries),end-begin};
        if (profile) {
            result.virtual_tree_seconds=std::chrono::duration<double>(built-started).count();
            result.moment_seconds=std::chrono::duration<double>(optimized-built).count();
            result.lookup_seconds=std::chrono::duration<double>(located-optimized).count();
            result.verification_seconds=std::chrono::duration<double>(Clock::now()-located).count();
        }
        return result;
    }
};

FrechetTreeIndex::FrechetTreeIndex(const std::string& tree):impl(std::make_shared<Impl>(tree)){}
FrechetTreeIndex::~FrechetTreeIndex()=default;
double FrechetTreeIndex::edge_length(Id id) const { return impl->edges[impl->edge_index(id)].length; }
std::vector<FrechetResult> FrechetTreeIndex::solve_batch(const std::vector<Id>& ids,const std::vector<double>& distal,const std::vector<double>& weights,const std::vector<Id>& offsets,const std::vector<Id>& queries,int threads,bool profile) const {
    if(threads<=0)throw std::invalid_argument("num_threads must be positive");
    if(ids.size()!=distal.size() || ids.size()!=weights.size() || offsets.size()!=queries.size()+1 || offsets.empty() || offsets.front()!=0 || offsets.back()!=ids.size())throw std::invalid_argument("Invalid batch buffers or offsets");
    std::vector<Point> points; points.reserve(ids.size());
    for(size_t i=0;i<ids.size();++i) {
        int e=impl->edge_index(ids[i]); double L=impl->edges[e].length,d=distal[i];
        if(!std::isfinite(d) || !std::isfinite(weights[i]) || weights[i]<=0)throw std::invalid_argument("Batch weights must be positive and coordinates finite");
        double tol=32*std::numeric_limits<double>::epsilon()*L;
        if(d < -tol || d>L+tol)throw std::invalid_argument("Distal length outside tree edge");
        d=std::max(0.0,std::min(L,d)); points.push_back({e,d,weights[i]});
    }
    for(size_t i=0;i<queries.size();++i)if(!queries[i] || offsets[i]>=offsets[i+1])throw std::invalid_argument("Empty consensus group");
    std::vector<FrechetResult> results(queries.size()); std::vector<std::string> errors(queries.size());
    if(queries.empty())return results;
    int workers=static_cast<int>(std::min<size_t>(threads,queries.size()));
    #pragma omp parallel for num_threads(workers) schedule(dynamic)
    for(size_t i=0;i<queries.size();++i) {
        try { results[i]=impl->solve(points,offsets[i],offsets[i+1],queries[i],profile); }
        catch(const std::exception& e) { errors[i]=e.what(); }
        catch(...) { errors[i]="Unknown solver error"; }
    }
    for(size_t i=0;i<errors.size();++i)if(!errors[i].empty())throw std::runtime_error("Consensus group "+std::to_string(i)+": "+errors[i]);
    return results;
}
