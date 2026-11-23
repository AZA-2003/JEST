"""Exact finite-candidate oracle, analytic cases, and production API integration."""
import csv
import json
import math
import random
from copy import deepcopy

import pytest
from typer.testing import CliRunner
from JEST import placement_consensus, subgenome_consensus
from JEST.JEST import app
from JEST.CTree import FrechetTreeIndex
from JEST.frechet import consensus
from test_frechet import make_newick


def solve(tree, points, alpha=1, index=None, threads=1):
    index = index or FrechetTreeIndex(tree)
    return index.exponential_mode_batch([e for e, d, w in points], [d for e, d, w in points],
        [w for e, d, w in points], [0, len(points)], [1], alpha, threads)[0]


def document(queries):
    return {"tree": "(tip:10{16777217})root:0{9};", "fields": ["distal_length", "edge_num", "like_weight_ratio", "pendant_length"],
            "placements": [{"n": [f"q{i}"], "p": [[d,16777217,w,2] for d,w in rows]} for i,rows in enumerate(queries)],
            "metadata": {"source":"test"}, "version":3}


def test_single_support_duplicates_and_precision():
    tree = "(tip:0.00000007{18446744073709551615}):0{7};"
    for alpha in [1e-300, 1, 1e300]:
        for points in [[(2**64-1,.0000000123456789,1)], [(2**64-1,.0000000123456789,w) for w in [.2,.3,.5]]]:
            result = solve(tree, points, alpha)
            assert result.edge_num == 2**64-1
            assert result.distal_length == .0000000123456789
            assert result.kernel_score == 1
            assert result.mean_squared_distance == 0
            assert result.kernel_tie_count == 1


def test_two_points_endpoints_not_midpoint_and_rescaling():
    tree = "(tip:10{20});"
    for scale in [1e-200, 1, 1e200]:
        result = solve(tree, [(20,2,scale), (20,8,2*scale)], .5)
        assert result.distal_length == 8
        assert result.kernel_score == pytest.approx((2+math.exp(-3))/3)
        assert result.kernel_tie_count == 1
    result = solve(tree, [(20,2,1), (20,8,1)], .5)
    assert result.distal_length == 2
    assert result.kernel_tie_count == 2
    assert result.kernel_score == pytest.approx((1+math.exp(-3))/2)
    assert result.kernel_score > math.exp(-1.5)


@pytest.mark.parametrize("alpha,distal,ties", [(.1,1,1), (2,0,3), (math.log(2),0,4)])
def test_star_center_or_tips(alpha,distal,ties):
    r = math.exp(-alpha)
    result = solve("(a:1{1},b:1{2},c:1{3});", [(1,0,1),(2,0,1),(3,0,1)], alpha)
    assert (result.edge_num,result.distal_length) == (1,distal)
    assert result.kernel_score == pytest.approx(max(r,(1+2*r*r)/3))
    assert result.kernel_tie_count == ties


def test_near_ties_compare_to_maximum_and_tiny_alpha():
    tree = "(a:1{30},b:1{20},c:1{10});"
    tolerance = solve(tree, [(30,0,1)]).kernel_tie_tolerance
    result = solve(tree, [(30,0,1/3),(20,0,1/3-.75*tolerance),(10,0,1/3-1.5*tolerance)],1e300)
    assert result.edge_num == 20
    assert result.kernel_tie_count == 2
    tiny = solve(tree, [(30,0,1),(20,0,1),(10,0,1)],1e-300)
    assert tiny.kernel_score == pytest.approx(1)
    assert tiny.kernel_tie_count == 4  # numerical ties, not proof of mathematical equality


def test_zero_components_positive_stem_and_repeated_alpha():
    tree = "((a:0{8},b:0{7}):0{6},c:1{9}):2{3};"
    index = FrechetTreeIndex(tree)
    first = solve(tree, [(8,0,1),(7,0,1)], .3,index)
    assert (first.edge_num,first.distal_length,first.kernel_score) == (3,0,1)
    for alpha in [.01,100,.01]:
        result = solve(tree, [(3,1.234567890123,2),(9,0,1)],alpha,index)
        assert result.edge_num == 3 and result.distal_length == 1.234567890123
        assert result.kernel_score == pytest.approx((2+math.exp(-alpha*2.234567890123))/3)
    all_zero = solve("(a:0{3},b:0{2}):0{1};", [(2,0,1),(3,0,2)],1e300)
    assert (all_zero.edge_num,all_zero.distal_length,all_zero.kernel_score) == (1,0,1)


def test_equal_query_rows_names_metadata_and_serialization(tmp_path):
    doc = document([[(0,1)],[(8,1),(10,1)]])
    out, diagnostics = consensus(doc,grouped=True,method="exponential_mode",kernel_alpha=1)
    assert out["placements"][0]["p"][0][3] == 0  # query A has mass 1/2
    assert diagnostics[0]["kernel_score"] == pytest.approx(.5+.25*math.exp(-8)+.25*math.exp(-10))
    assert diagnostics[0]["query_count"] == 2
    altered = deepcopy(doc)
    altered["placements"][1]["p"] = [[8,16777217,1e100,2],[10,16777217,.5e100,2],[10,16777217,.5e100,2]]
    other, diag = consensus(altered,grouped=True,method="exponential_mode",kernel_alpha=1,num_threads=4)
    assert other["placements"] == out["placements"]
    assert diag[0]["kernel_score"] == diagnostics[0]["kernel_score"]
    assert out["tree"] == doc["tree"]
    assert "frechet" not in out["metadata"]
    assert out["metadata"]["exponential_mode"]["method"] == "exponential_mode"
    assert out["metadata"]["source"] == "test"
    path = tmp_path/'queries.jplace';path.write_text(json.dumps(doc))
    cli = CliRunner().invoke(app,["subgenome-consensus","-i",str(path),"-w","-e","-o",str(tmp_path),"--method","exponential_mode","--kernel-alpha","1"])
    assert cli.exit_code == 0, cli.output+str(cli.exception)
    saved = json.loads((tmp_path/'queries_subgenome_consensus.jplace').read_text())
    assert saved == out
    with (tmp_path/'queries_subgenome_exponential_mode.tsv').open() as stream:
        rows = list(csv.DictReader(stream,delimiter='\t'))
    assert float(rows[0]['kernel_score']) == diagnostics[0]['kernel_score']
    assert not (tmp_path/'queries_subgenome_frechet.tsv').exists()
    detailed = subgenome_consensus(str(path),False,method="exponential_mode",kernel_alpha=1,return_diagnostics=True)
    assert detailed['diagnostics'] == diagnostics
    record = doc['placements'][0];record.pop('n');record['nm']=[['A_one',3],['B_one',10]]
    doc['placements'][1]['n']=['A_two','B_two']
    path.write_text(json.dumps(doc))
    placements = placement_consensus(str(path),False,method='exponential_mode',kernel_alpha=1)
    assert placements[0]['nm'] == record['nm']
    grouped, _ = subgenome_consensus(str(path),False,method='exponential_mode',kernel_alpha=1,genome_map=r'(.*)_(.*)')
    assert [p['n'] for p in grouped] == [['A'],['B']]
    cli = CliRunner().invoke(app,["placement-consensus","-i",str(path),"-w","-o",str(tmp_path),"--method","exponential_mode","--kernel-alpha","1"])
    assert cli.exit_code == 0,cli.output+str(cli.exception)


@pytest.mark.parametrize('alpha',[None,0,-1,float('inf'),float('nan'),True])
def test_kernel_parameter_validation(alpha,tmp_path):
    doc=document([[(0,1)]])
    with pytest.raises(ValueError):consensus(doc,method='exponential_mode',kernel_alpha=alpha)
    path=tmp_path/'input.jplace';path.write_text(json.dumps(doc))
    for method in ['legacy','frechet']:
        with pytest.raises(ValueError):placement_consensus(str(path),False,method=method,kernel_alpha=1)
    if alpha is not None and alpha is not True:
        with pytest.raises(ValueError):solve(doc['tree'],[(16777217,0,1)],alpha)


def test_filter_rejection_and_batch_validation(tmp_path):
    doc=document([[(0,1)]]);path=tmp_path/'input.jplace';path.write_text(json.dumps(doc))
    for options in [dict(pre_filter=True),dict(post_filter=True),dict(filter_queries=True),dict(pre_alpha=.2),dict(random_dist_path='old.txt')]:
        with pytest.raises(ValueError,match='not yet calibrated'):
            subgenome_consensus(str(path),False,method='exponential_mode',kernel_alpha=1,**options)
    index=FrechetTreeIndex(doc['tree'])
    assert len(index.exponential_mode_batch([],[],[],[0],[],1,4)) == 0
    for offsets,counts,threads in [([0,0],[1],1),([0,1],[0],1),([0,1],[1],0)]:
        with pytest.raises(ValueError):index.exponential_mode_batch([16777217],[0],[1],offsets,counts,1,threads)


def oracle(parent,lengths,points,alpha):
    """Subdivide the FULL original tree and score every vertex by graph walks."""
    n=len(parent); graph=[[] for _ in parent]; observation_nodes=[]
    occupied={e:{} for e in range(1,n)}
    for e,d,w in points:
        t=lengths[e]-d
        if d==0: v=e
        elif d==lengths[e]: v=parent[e]
        else:
            if t not in occupied[e]:
                occupied[e][t]=len(graph);graph.append([])
            v=occupied[e][t]
        observation_nodes.append((v,w))
    for e in range(1,n):
        chain=sorted([(0,parent[e]),(lengths[e],e)]+list(occupied[e].items()))
        for (a,u),(b,v) in zip(chain,chain[1:]):
            graph[u].append((v,b-a));graph[v].append((u,b-a))
    def distances(start):
        ds={start:0};stack=[start]
        while stack:
            u=stack.pop()
            for v,d in graph[u]:
                if v not in ds:ds[v]=ds[u]+d;stack.append(v)
        return ds
    all_distances=[distances(v) for v in range(len(graph))]
    W=math.fsum(w for _,w in observation_nodes)
    scores=[math.fsum(w*math.exp(-alpha*ds[v]) for v,w in observation_nodes)/W for ds in all_distances]
    interior_locations = {v:(e,t) for e,locations in occupied.items() for t,v in locations.items()}
    def membership_distance(edge,distal,tolerance):
        best=max(scores)
        distances_to_winners = []
        for v,ds in enumerate(all_distances):
            if best-scores[v] > tolerance:
                continue
            d = min(ds[edge]+distal,ds[parent[edge]]+lengths[edge]-distal)
            if v in interior_locations and interior_locations[v][0] == edge:
                d = min(d,abs(interior_locations[v][1]-(lengths[edge]-distal)))
            distances_to_winners.append(d)
        return min(distances_to_winners)
    return max(scores),membership_distance


def test_random_exhaustive_original_tree_oracle():
    rng=random.Random(224891)
    for trial in range(2000):
        n=rng.randrange(2,16)
        parent=[0]+[rng.randrange(i) if trial%3 else i-1 for i in range(1,n)]
        lengths=[0]+[rng.choice([0,1e-5,.1,1,100])*rng.uniform(.5,2) for _ in range(n-1)]
        points=[]
        for _ in range(rng.randrange(1,16)):
            e=rng.randrange(1,n)
            points.append((e,rng.choice([0,lengths[e],lengths[e]*rng.random()]),10**rng.uniform(-5,2)))
        alpha=10**rng.uniform(-6,4)
        result=solve(make_newick(parent,lengths),points,alpha)
        best,distance=oracle(parent,lengths,points,alpha)
        assert result.kernel_score==pytest.approx(best,rel=1e-11,abs=2e-12),trial
        assert distance(result.edge_num,result.distal_length,max(2e-12,result.kernel_tie_tolerance))<1e-9*max(1,sum(lengths)),trial


def test_root_subdivision_invariance_and_high_degree_parallel():
    points=[(7,0,.7),(8,0,.2),(9,0,.1)]
    a=solve('(a:10{7},b:3{8},c:2{9});',points,.2)
    b=solve('((a:4{7}):6{17},b:3{8},c:2{9});',points,.2)
    c=solve('((b:3{8},c:2{9}):10{27})a;',[(27,10,.7),(8,0,.2),(9,0,.1)],.2)
    assert (a.edge_num,a.distal_length)==(7,0)
    assert (b.edge_num,b.distal_length)==(7,0)
    assert (c.edge_num,c.distal_length)==(27,10)
    assert a.kernel_score==pytest.approx(b.kernel_score)
    assert a.kernel_score==pytest.approx(c.kernel_score)
    n=1000
    index=FrechetTreeIndex('('+','.join(f'x{i}:1{{{i}}}' for i in range(n))+');')
    params=(list(range(n))*8,[0]*(8*n),[1]*(8*n),list(range(0,8*n+1,n)),[1]*8,.1)
    serial=index.exponential_mode_batch(*params,1)
    parallel=index.exponential_mode_batch(*params,4)
    for x,y in zip(serial,parallel):
        assert (x.edge_num,x.distal_length,x.kernel_score,x.kernel_tie_count)==(y.edge_num,y.distal_length,y.kernel_score,y.kernel_tie_count)
        assert x.distal_length==1 and x.kernel_tie_count==1


def test_long_root_short_local_segments_and_bounded_score():
    tree = '(a:0.00001{1},b:0.00002{2}):100000000{3};'
    result = solve(tree,[(1,0,1),(2,0,1)],1e5)
    assert result.kernel_score == pytest.approx((1+math.exp(-3))/2,rel=1e-14)
    assert result.kernel_tie_count == 2
    extreme = solve('(a:1{1},b:1{2});',[(1,0,1e308),(2,0,1e308)],1.7976931348623157e308)
    assert extreme.kernel_score == .5 and extreme.kernel_tie_count == 2
    doc = {'tree':'(a:1e200{1});','fields':['edge_num','like_weight_ratio','distal_length'],
           'placements':[{'n':['q'],'p':[[1,1,0],[1,1,1e200]]}]}
    output,diag = consensus(doc,method='exponential_mode',kernel_alpha=1)
    assert diag[0]['kernel_score'] == .5
    assert diag[0]['mean_squared_distance'] is None
    json.dumps(output,allow_nan=False)
