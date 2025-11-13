# N-Gram Generation Toolkit

A high-performance n-gram generation library with Rust core and Polars plugin integration.

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

## Quick Start


```rust
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
