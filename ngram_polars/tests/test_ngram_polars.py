import polars as pl
import sys
from pathlib import Path
from polars.testing import assert_series_equal, assert_frame_equal
import polars.plugins


# Find the compiled plugin library
try:
    import ngram_polars
    # The .so/.pyd file should be alongside the __init__.py
    module_dir = Path(ngram_polars.__file__).parent
    
    # # Try different possible names
    # if sys.platform == "win32":
    #     lib_name = "ngram_polars.pyd"
    # elif sys.platform == "darwin":
    #     lib_name = "ngram_polars.so"
    # else:
    #     lib_name = "ngram_polars.so"
    
    PLUGIN_PATH = module_dir 
    # PLUGIN_PATH = module_dir / lib_name
    
    # # Fallback: look in parent directory
    # if not PLUGIN_PATH.exists():
    #     PLUGIN_PATH = module_dir.parent / lib_name
    
    # if not PLUGIN_PATH.exists():
    #     PLUGIN_PATH = Path(".venv/lib/python3.12/site-packages/ngram_polars/ngram_polars.cpython-312-darwin.so")
   
    # if not PLUGIN_PATH.exists():
    #     raise FileNotFoundError(f"Could not find plugin at {PLUGIN_PATH}")
        
except ImportError:
    raise ImportError("ngram_polars not installed. Run: maturin develop --release")

def test_basic_bigrams():
    """Test basic bigram generation"""
    df = pl.DataFrame({
        "words": [
            ["the", "quick", "brown", "fox"],
            ["hello", "world"],
        ]
    })
    result = df.with_columns(
        pl.col("words").register_plugin(
            lib=PLUGIN_PATH,
            symbol="ngrams",
            kwargs={"n_range": [2], "delimiter": " "},
        ).alias("ngrams")
    )

    print("Test: Basic Bigrams")
    print(result)
    expected_result = pl.DataFrame({
        "ngrams":[
            ["the quick", "quick brown", "brown fox"],
            ["hello world"]
        ]
    }).to_series()
    assert_series_equal(result.to_series(1), expected_result)
    print("✓ Passed\n")


def test_trigrams():
    """Test trigram generation"""
    df = pl.DataFrame({
        "words": [["a", "b", "c", "d"]]
    })
    
    result = df.with_columns(
        pl.col("words").register_plugin(
            lib=PLUGIN_PATH,
            symbol="ngrams",
            kwargs={"n_range": [3], "delimiter": " "},
        ).alias("ngrams")
    )
    
    print("Test: Trigrams")
    print(result)
    assert_series_equal(result.select('ngrams').to_series(0), pl.Series(name='ngrams', values=[["a b c", "b c d"]]))
    print("✓ Passed\n")


def test_multiple_n_values():
    """Test multiple n-gram sizes at once"""
    df = pl.DataFrame({
        "words": [["x", "y", "z"]]
    })
    
    result = df.with_columns(
        pl.col("words").register_plugin(
            lib=PLUGIN_PATH,
            symbol="ngrams",
            kwargs={"n_range": [1, 2, 3], "delimiter": " "},
        ).alias("ngrams")
    )
    
    print("Test: Multiple N-values (1, 2, 3)")
    print(result)
    assert_series_equal(result.select('ngrams').to_series(),
                        pl.Series('ngrams', [["x", "y", "z", "x y", "y z", "x y z"]], dtype=pl.List(pl.String)))
    # expected = ["x", "y", "z", "x y", "y z", "x y z"]
    # assert result["ngrams"][0] == expected
    print("✓ Passed\n")


def test_custom_delimiter():
    """Test custom delimiter"""
    df = pl.DataFrame({
        "words": [["foo", "bar", "baz"]]
    })
    
    result = df.with_columns(
        pl.col("words").register_plugin(
            lib=PLUGIN_PATH,
            symbol="ngrams",
            kwargs={"n_range": [2], "delimiter": "_"},
        ).alias("ngrams")
    )
    
    print("Test: Custom Delimiter (_)")
    print(result)
    assert_series_equal(result.select('ngrams').to_series(),
                        pl.Series('ngrams', [["foo_bar", "bar_baz"]], dtype=pl.List(pl.String)))
    print("✓ Passed\n")


def test_default_delimiter():
    """Test default delimiter (space)"""
    df = pl.DataFrame({
        "words": [["hello", "world"]]
    })
    
    result = df.with_columns(
        pl.col("words").register_plugin(
            lib=PLUGIN_PATH,
            symbol="ngrams",
            kwargs={"n_range": [2]},
        ).alias("ngrams")
    )
    
    print("Test: Default Delimiter (space)")
    print(result)
    assert_series_equal(result.select('ngrams').to_series(),
                        pl.Series('ngrams', [["hello world"]], dtype=pl.List(pl.String)))
    print("✓ Passed\n")


def test_empty_list():
    """Test handling of empty lists"""
    df = pl.DataFrame({
        "words": [[]]
    })
    
    result = df.with_columns(
        pl.col("words").register_plugin(
            lib=PLUGIN_PATH,
            symbol="ngrams",
            kwargs={"n_range": [2], "delimiter": " "},
        ).alias("ngrams")
    )
    
    print("Test: Empty List")
    print(result)
    assert_series_equal(result.select('ngrams').to_series(),
                        pl.Series('ngrams', [[]], dtype=pl.List(pl.String)))
    print("✓ Passed\n")


def test_single_word():
    """Test list with single word"""
    df = pl.DataFrame({
        "words": [["solo"]]
    })
    
    result = df.with_columns(
        pl.col("words").register_plugin(
            lib=PLUGIN_PATH,
            symbol="ngrams",
            kwargs={"n_range": [2], "delimiter": " "},
        ).alias("ngrams")
    )
    
    print("Test: Single Word (no bigrams possible)")
    print(result)
    assert_series_equal(result.select('ngrams').to_series(),
                        pl.Series('ngrams', [[]], dtype=pl.List(pl.String)))
    print("✓ Passed\n")


def test_unigrams():
    """Test unigram generation (should return individual words)"""
    df = pl.DataFrame({
        "words": [["alpha", "beta", "gamma"]]
    })
    
    result = df.with_columns(
        pl.col("words").register_plugin(
            lib=PLUGIN_PATH,
            symbol="ngrams",
            kwargs={"n_range": [1], "delimiter": " "},
        ).alias("ngrams")
    )
    
    print("Test: Unigrams")
    print(result)
    assert_series_equal(result.select('ngrams').to_series(),
                        pl.Series('ngrams', [["alpha", "beta", "gamma"]]))
    print("✓ Passed\n")


def test_multiple_rows():
    """Test with multiple rows"""
    df = pl.DataFrame({
        "id": [1, 2, 3],
        "words": [
            ["the", "quick", "brown"],
            ["lazy", "dog"],
            ["jumps", "over", "the", "fence"]
        ]
    })
    
    result = df.with_columns(
        pl.col("words").register_plugin(
            lib=PLUGIN_PATH,
            symbol="ngrams",
            kwargs={"n_range": [2], "delimiter": " "},
        ).alias("ngrams")
    )
    
    print("Test: Multiple Rows")
    print(result)
    assert_series_equal(result.select('ngrams').to_series(),
                        pl.Series('ngrams', 
                                  [["the quick", "quick brown"],
                                   ["lazy dog"],
                                   ["jumps over", "over the", "the fence"]]))
    print("✓ Passed\n")


def test_with_lazy_api():
    """Test with Polars lazy API"""
    df = pl.DataFrame({
        "words": [["hello", "beautiful", "world"]]
    })
    
    result = (
        df.lazy()
        .with_columns(
            pl.col("words").register_plugin(
                lib=PLUGIN_PATH,
                symbol="ngrams",
                kwargs={"n_range": [2, 3], "delimiter": " "},
            ).alias("ngrams")
        )
        .collect()
    )
    
    print("Test: Lazy API")
    print(result)
    assert_series_equal(result.select('ngrams').to_series(), pl.Series('ngrams', [["hello beautiful", "beautiful world", "hello beautiful world"]]))
    print("✓ Passed\n")


def test_with_filter():
    """Test chaining with other Polars operations"""
    df = pl.DataFrame({
        "category": ["A", "B", "A"],
        "words": [
            ["one", "two", "three"],
            ["four", "five"],
            ["six", "seven", "eight"]
        ]
    })

    expected_results = pl.DataFrame({
        'category': ['A', 'A'],
        "words": [
            ["one", "two", "three"],
            ["six", "seven", "eight"]
        ],
        "ngrams": [
            ["one-two", "two-three"],
            ["six-seven", "seven-eight"]
        ]
 
    })
    
    result = (
        df.filter(pl.col("category") == "A")
        .with_columns(
            pl.col("words").register_plugin(
                lib=PLUGIN_PATH,
                symbol="ngrams",
                kwargs={"n_range": [2], "delimiter": "-"},
            ).alias("ngrams")
        )
    )
    
    print("Test: Chaining with Filter")
    print(result)
    assert_frame_equal(result, expected_results)
    print("✓ Passed\n")


def test_large_n_range():
    """Test with n larger than list length"""
    df = pl.DataFrame({
        "words": [["a", "b"]]
    })
    
    result = df.with_columns(
        pl.col("words").register_plugin(
            lib=PLUGIN_PATH,
            symbol="ngrams",
            kwargs={"n_range": [1, 2, 5, 10], "delimiter": " "},
        ).alias("ngrams")
    )
    
    print("Test: N-values larger than list (should be ignored)")
    print(result)
    # Should only generate unigrams and bigrams
    assert_series_equal(result.select('ngrams').to_series(0), pl.Series(name='ngrams', values=[["a", "b", "a b"]]))
    print("✓ Passed\n")


def test_groupby_aggregation():
    """Test with groupby aggregation"""
    df = pl.DataFrame({
        "group": ["A", "A", "B", "B"],
        "words": [
            ["hello", "world"],
            ["foo", "bar"],
            ["alpha", "beta", "gamma"],
            ["one", "two"]
        ]
    })

    expected_results = pl.DataFrame({
        "group": [ "A", "B", ],
        "ngrams": [
            ["hello world", "foo bar"],
            ["alpha beta", "beta gamma", 'one two'],
        ]
    }).sort(by=['group','ngrams'])
    
    result = df.with_columns(
        pl.col("words").register_plugin(
            lib=PLUGIN_PATH,
            symbol="ngrams",
            kwargs={"n_range": [2], "delimiter": " "},
        ).alias("ngrams")
    ).group_by("group").agg(
        pl.col("ngrams").flatten()
    ).sort(by=['group','ngrams'])
    

    print("Test: GroupBy Aggregation")
    print(result)
    assert_frame_equal(result, expected_results)
    print("✓ Passed\n")


def test_package_info():
    """Test that package is properly imported"""
    print("Test: Package Info")
    print("Expr has ngrams:", hasattr(pl.Expr, "ngrams"))
    print("ngram_polars attrs:", dir(ngram_polars))
    print("ngram_polars.ngrams exists?:", hasattr(ngram_polars, "ngrams"))
    print("ngram_polars.ngrams repr:", getattr(ngram_polars, "ngrams", None))
    print("✓ Passed\n")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running N-gram Polars Plugin Tests")
    print("=" * 60 + "\n")
    
    tests = [
        test_package_info,
        test_basic_bigrams,
        test_trigrams,
        test_multiple_n_values,
        test_custom_delimiter,
        test_default_delimiter,
        test_empty_list,
        test_single_word,
        test_unigrams,
        test_multiple_rows,
        test_with_lazy_api,
        test_with_filter,
        test_large_n_range,
        test_groupby_aggregation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {test.__name__}")
            print(f"  Error: {e}\n")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {test.__name__}")
            print(f"  Error: {e}\n")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)