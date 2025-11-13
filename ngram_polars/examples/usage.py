import polars as pl
from ngram_polars import ngrams

def main():
    # Create sample data
    df = pl.DataFrame({
        "id": [1, 2, 3],
        "words": [
            ["the", "quick", "brown", "fox"],
            ["hello", "world"],
            ["rust", "polars", "plugin", "example", "demo"]
        ]
    })
    
    print("Original DataFrame:")
    print(df)
    print()
    
    # Example 1: Basic bigrams with space delimiter
    result1 = df.with_columns(
        bigrams = ngrams(pl.col("words"), [2], " ")
    )
    
    print("Bigrams (n=2):")
    print(result1)
    print()
    
    # Example 2: Multiple n-grams (bigrams and trigrams)
    result2 = df.with_columns(
        ngrams_2_3 = ngrams(pl.col("words"), [2, 3], " ")
    )
    
    print("Bigrams + Trigrams (n=[2, 3]):")
    print(result2)
    print()
    
    # Example 3: Custom delimiter
    result3 = df.with_columns(
        bigrams_underscore = ngrams(pl.col("words"), n_range=[2], delimiter= "_")
    )
    
    print("Bigrams with underscore delimiter:")
    print(result3)
    print()
    
    # Example 4: Range of n-grams (1 to 4)
    result4 = df.with_columns(
        ngrams_1_to_4 = ngrams(pl.col("words"), n_range=[1, 2, 3, 4], delimiter=" | ")
    )
    
    print("N-grams from 1 to 4 with custom delimiter:")
    print(result4)
    print()
    
    # Example 5: Using in lazy mode
    lazy_result = df.lazy().with_columns(
        ngrams_lazy = ngrams(pl.col("words"), [2, 3], " -> ")
    ).collect()
    
    print("Lazy evaluation result:")
    print(lazy_result)

if __name__ == "__main__":
    main()