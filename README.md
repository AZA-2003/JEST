# JEST
The J-place Evaluation and Scoring Tool (JEST) is a python3-based tool to evaluate genome/gene placement results (saved as a jplace). The tool consists of two features:
- placement uncertainty score: measures confidence of a query placement when compared to random placement. This is applicable when the algorithm provides multiple placements and the ground truth is unknown.
- placement error: measures the distance (both in number of edges on the tree and the total branch length) between the query placement and its ground truth.

## Setup
JEST can be installed using `pip`:
`pip install JEST`
## Usage
