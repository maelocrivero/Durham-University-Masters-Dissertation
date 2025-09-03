# probabilitylib Library

A library designed to bridge the gap between Data Science and Probability Theory.

This library treats pandas DataFrames as probability spaces. This means treating the columns within the DataFrames as random variables. The functions within the library allow for probabilistic analysis of these variables. The library also includes methods allowing the functions to be run on datasets that are too large to fit in memory.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/maelocrivero/probabilitylib.git
cd dissertation
pip install -r requirements.txt
```


## Usage
Import probabilitylib and the ProbabiltiySpace class

```
import probabilitylib as pl
from probabilitylib import ProbabilitySpace
import pandas as pd

# Example DataFrame
df = pd.DataFrame({"X": [1, 2, 3], "Y": [4, 5, 6]})
ps = ProbabilitySpace(df)

# Compute expected mean
print("E[X] =", pl.expected_mean(ps, "X")

# Compute variance
print("Var[Y] =", pl.variance(ps, "Y"))
```


