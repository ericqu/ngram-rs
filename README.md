# N-Gram Generation Toolkit

A high-performance n-gram generation library with Rust core and Polars plugin integration.

## Project Structure
```
.
├── ngram_rs/ # Core Rust n-gram library
├── ngram_polars/ # Polars plugin for Python
└── Cargo.toml # Workspace configuration
```


## Features

- **Blazing Fast**: Optimized Rust implementation for n-gram generation
- **Memory Efficient**: Uses `Cow` (Copy-on-Write) for minimal allocations
- **Flexible N-Ranges**: Generate n-grams for multiple values of n simultaneously
- **Custom Delimiters**: Support for any string delimiter between tokens
- **Polars Integration**: Seamless integration with Polars DataFrames
- **Iterator Support**: Lazy n-gram generation for memory-constrained environments

## Components

### ngram_rs (Core Library)
The core Rust library providing:
- Three different APIs for various use cases
- Optimized implementations for common cases (unigrams, bigrams)
- Iterator-based lazy generation

### ngram_polars (Python Plugin)
A Polars plugin that exposes n-gram functionality to Python with:
- Native Polars expression integration
- Support for both eager and lazy evaluation
- Element-wise operations on string lists

## Quick Start

```python
# Python usage
import polars as pl
from ngram_polars import ngrams

df = pl.DataFrame({
    "words": [["the", "quick", "brown", "fox"]]
})

result = df.with_columns(
    ngrams=ngrams(pl.col("words"), n_range=[1, 2, 3], delimiter=" ")
)
```

```rust
// Rust usage
use ngram_rs::generate_ngrams;

let words = vec!["the", "quick", "brown", "fox"]
    .into_iter()
    .map(String::from)
    .collect::<Vec<_>>();

let ngrams = generate_ngrams(&words, &[1, 2, 3], Some(" "));
```

## Performance
The library is optimized for:

- Minimal memory allocations through Cow<str>
- Specialized implementations for unigrams and bigrams
- Efficient windowing algorithms for higher-order n-grams
- Zero-copy operations where possible

## Changes
0.1.2:  Updated to Rust 1.93.1 and Polars 0.53.0 (dropped python 3.9 added 3.14)
