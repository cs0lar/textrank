# textrank

Python implementation of the textrank algorithm (mihalcea, tarau 2004)

## Installation

`pip install git+https://github.com/cs0lar/textrank.git`

## Usage

### Ranking
Use the `rank()` method to perform a full ranking of all the keywords in the input text. The `rank()` method return keyword-score pairs sorted in descending order:

```python
from texrank.textrank import TexRank

ranker = TextRank( N=2 )

text = 'Low index linear systems with those properly stated leading terms are considered in detail. In particular, it is asked whether a numerical integration method applied to the original system reaches the inherent regular ODE without conservation, i.e., whether the discretization and the decoupling commute in some sense. In general one cannot expect this commutativity so that additional difficulties like strong stepsize restrictions may arise. Moreover, abstract differential algebraic equations in infinite-dimensional Hilbert spaces are introduced, and the index notion is generalized to those equations. In particular, partial differential algebraic equations are considered in this abstract formulation'

print ( ranker.rank( text ) )
```
### Keywords

Use the `keywords()` method to return the top `T` ranked keywords from the input text. If `T` is not specified, this method sets `T` to be one-third of the vertices of the textrank graph.

```python
print ( ranker.keywords( text, T=10 ) )

```

### Multi-word keywords

Use the `multikeywords()` method to return the top `T` ranked multi-word keywords from the input text:

```python
print ( ranker.multikeywords( text ) )

```

### Graph

Sometimes it is useful to inspect the graph that `TextRank` generates for a given text. To visualise a textrank graph use the `graph()` method with the `plot` argument set to `True`. It uses `matplotlib` internally to draw the network therefore in order to show the visualisation the `matplotlib.pyplot` module must be loaded and the `show()` function invoked. It takes as arguments the tokens extracted from the input text and the subset of these tokens that is to constitute the graph's vertices.
Use the `preprocess()` method to obtain these.

```python
import matplotlib.pyplot as plt

tokens, vertices = ranker.preprocess( text )

ranker.graph( vertices, tokens, plot=True )

plt.show()

```

