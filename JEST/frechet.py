"""Backward-compatible imports for the shared :mod:`JEST.tree_consensus` API."""
from .tree_consensus import consensus, run
from .CTree import TreeConsensusIndex as FrechetTreeIndex

__all__ = ["consensus", "run", "FrechetTreeIndex"]
