# JEST
The Jplace Evaluation and Scoring Tool (JEST) is a Python3-based tool for evaluating and summarizing placement results (saved in a JSON-format known as a jplace file) allowing for post-placement analysis.   

The tool consists of three features:
- **placement uncertainty score**  
measures confidence of a query placement when compared to randomly placing on a tree. This is applicable when the algorithm provides multiple placements and the ground truth is unknown. This also includes removing queries that have high uncertainty scores, providing a jplace file with these queries excluded.
- **placement-based consensus**  
summarizes the placements of a given query to a single placement on the tree-- also known as the consensus placement. This also includes measuring the distance between the consensus placement and a ground truth given a ground truth label and (optionally) a tree. This is useful for leave-out experiments where the tree is the same as the jplace with the ground truth node.  
Note that this algorithm handles each query independently of the other.
- **gene-based consensus**  
This works similarly to placement-based consensus but assumes that all queries in the file belong to the same ground truth label; this results in pooling all the given placements and summarzies to a single consensus. This also includes measuring the distance between the consensus placement and a ground truth given a ground truth label and (optionally) a tree. This is useful in cases where each query in the jplace file is a gene that all belong to a genome.

While JEST is a Python library, its main implementations wrap around functions written in C++ to allow for speed and multi-threading-support. As such, [CompactTree](https://github.com/niemasd/CompactTree) is used for tree parsing and manipulation thereby only supporting the [Newick tree format](https://en.wikipedia.org/wiki/Newick_format) with no complex annotations (only node labels and edge lengths).

## Setup
JEST can be installed using `pip`:  
`pip install JEST`
## Usage (with a toy example)
