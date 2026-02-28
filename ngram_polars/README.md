# ngram_polars - N-Gram Generation for Polars

A high-performance Polars plugin for generating n-grams from text data in Python.

## Installation

```bash
pip install ngram-polars
```

# Basic example
```python
import polars as pl
from ngram_polars import ngrams

df = pl.DataFrame({
    "id": [1, 2],
    "words": [
        ["the", "quick", "brown", "fox"],
        ["hello", "world"]
    ]
})

# Generate bigrams
result = df.with_columns(
    bigrams=ngrams(pl.col("words"), n_range=[2])
)
```

# more advanced examples

```python
# Multiple n-gram sizes
df.with_columns(
    multi_ngrams=ngrams(pl.col("words"), n_range=[1, 2, 3])
)

# Custom delimiter
df.with_columns(
    underscored=ngrams(pl.col("words"), n_range=[2], delimiter="_")
)

# Lazy evaluation
(df.lazy()
   .with_columns(
       ngrams=ngrams(pl.col("words"), n_range=[2, 3])
   )
   .collect()
)
```

## API Reference

`ngrams(expr, n_range, delimiter)`
Generate n-grams from a list of strings.

### Parameters:
- `expr: IntoExpr` - Polars expression representing a list of strings
- `n_range: list[int]` - List of n-gram sizes to generate (default: [1])
- `delimiter: str` - String delimiter between words (default: " ")
### Returns:
- `pl.Expr` - Expression that generates lists of n-gram strings

### Behavior:
- Returns a new list column containing all generated n-grams
- Works element-wise on list columns
- Changes the length of the output (each input list produces a new list of n-grams)
- Supports both eager and lazy evaluation

### Performance Tips
- Use Lazy Evaluation: For large datasets, use lazy evaluation to optimize query planning
- Batch N-Gram Sizes: Generate multiple n-gram sizes in one call when possible
- Choose Appropriate N-Range: Only generate the n-gram sizes you actually need

### Requirements
- Python 3.10 -> 3.14
- Polars requirement are not fully tested, tested on the latest version.
- Compatible with both eager and lazy Polars APIs

## Changes
0.1.2:  Updated to Rust 1.93.1 and Polars 0.53.0 (dropped python 3.9 added 3.14)
