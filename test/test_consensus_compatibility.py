"""Public aliases keep existing scripts working after shared-code renaming."""
from JEST import CTree
from JEST import frechet, tree_consensus


def test_public_python_names_remain_aliases():
    assert CTree.FrechetTreeIndex is CTree.TreeConsensusIndex
    assert CTree.FrechetResult is CTree.ConsensusResult
    assert CTree.FrechetPoint is CTree.TreePoint
    assert CTree.FrechetResults is CTree.ConsensusResults
    assert CTree.FrechetPoints is CTree.TreePoints
    assert frechet.consensus is tree_consensus.consensus
    assert frechet.run is tree_consensus.run
    tree = '(tip:2{7});'
    old = CTree.FrechetTreeIndex(tree)
    new = CTree.TreeConsensusIndex(tree)
    arguments = ([7, 7], [0, 2], [1, 1], [0, 2], [1], 1)
    for index in (old, new):
        result = index.solve_batch(*arguments)[0]
        assert isinstance(result, CTree.ConsensusResult)
        assert isinstance(result, CTree.FrechetResult)
        assert result.distal_length == 1
        assert index.exponential_mode_batch(*arguments[:-1], 1, 1)[0].kernel_tie_count == 2
