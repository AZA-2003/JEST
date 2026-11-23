# JEST
The Jplace Evaluation and Scoring Tool (JEST) is a Python3-based tool for evaluating and summarizing placement results (saved in a JSON-format known as a jplace file) allowing for post-placement analysis.   

The tool consists of three main features:
- **placement uncertainty score**  
measures confidence of a query placement when compared to randomly placing on a tree. This is applicable when the algorithm provides multiple placements and the ground truth is unknown. This also includes removing queries that have high uncertainty scores, providing a jplace file with these queries excluded.
- **placement-based consensus**  
summarizes the placements of a given query to a single placement on the tree-- also known as the consensus placement.
Note that this algorithm handles each query independently of the other.
- **subgenome-based consensus**  
This works similarly to placement-based consensus but pools different queries, based on a mapping, into a single consensus. This is useful in cases where each query in the jplace file is a subgenome that belongs to some genome.

Other features include measuring the sum of distances between a true placement and the query's placements weighed by their likelihood ratio. 
For detialed documentation and tutorial, please refer to the [Wiki](https://github.com/AZA-2003/JEST/wiki)

While JEST is a Python library, its main implementations wrap around functions written in C++ to allow for speed and multi-threading-support. As such, [CompactTree](https://github.com/niemasd/CompactTree) is used for tree parsing and manipulation thereby only supporting the [Newick tree format](https://en.wikipedia.org/wiki/Newick_format) with no complex annotations (only node labels and edge lengths).

## Setup
JEST can be installed using `pip` under the name JEST-Phylo:  
```bash
pip install JEST-Phylo
```
While it is recommended to use `pip` for installation, it is possible to set up JEST by cloning the repository and running the following steps:
```bash
cd JEST/JEST/
bash compile_CTree.sh $CONDA_PREFIX/include/python*
export PYTHONPATH=$PYTHONPATH:<absolute path to github clone>
```
After setup, JEST can be used in any Python3 file by importing as follows:
```Python3
import JEST
```
To make use of JEST as a CLI command, install `typer` in your environment:
```bash
pip install typer
```
To test out the CLI:
```bash
JEST --help
```
## Usage (with a toy example)



## Continuous Fréchet consensus

Use `method="frechet"` in `placement_consensus` or `subgenome_consensus`, or
`--method frechet` in the CLI, to minimize expected squared attachment distance
with equal total weight per query. The default remains `legacy`.

See [Fréchet consensus usage, conventions, and build instructions](docs/frechet_consensus.md).
Rebuild the SWIG extension before using the new method.

For optional Fréchet subgenome filtering, use `--pre-filter` (dispersion around
each query's own consensus) and/or `--post-filter` (distance from its group
consensus). Both are off by default; `--pre-alpha` and `--post-alpha` control
their separate null-percentile cutoffs. See the usage guide for scoring and
retained-query output semantics.

For exponential proximity modes, use `--method exponential_mode --kernel-alpha A`
with an explicit positive decay rate. This method shares equal-query weighting
and the tree index with Fréchet consensus, but can return multiple numerical
modes (resolved deterministically) and has its own `kernel_score` diagnostic.
See [exponential-mode usage and numerical conventions](docs/exponential_mode.md).
