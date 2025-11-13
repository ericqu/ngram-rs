import polars as pl
from polars.testing import assert_series_equal, assert_frame_equal
from ngram_polars import ngrams


def test_basic_bigrams():
    df = pl.DataFrame(
        {
            "words": [
                ["the", "quick", "brown", "fox"],
                ["hello", "world"],
            ]
        }
    )
    result = df.with_columns(ngrams=ngrams(pl.col("words"), n_range=[2], delimiter=" "))
    expected = pl.Series(
        "ngrams", [["the quick", "quick brown", "brown fox"], ["hello world"]]
    )
    assert_series_equal(result["ngrams"], expected)


def test_trigrams():
    df = pl.DataFrame({"words": [["a", "b", "c", "d"]]})
    result = df.with_columns(ngrams=ngrams(pl.col("words"), n_range=[3]))
    expected = pl.Series("ngrams", [["a b c", "b c d"]])
    assert_series_equal(result["ngrams"], expected)


def test_multiple_n_values():
    df = pl.DataFrame({"words": [["x", "y", "z"]]})
    result = df.with_columns(ngrams(pl.col("words"), n_range=[1, 2, 3]).alias("ngrams"))
    expected = pl.Series(
        "ngrams", [["x", "y", "z", "x y", "y z", "x y z"]], dtype=pl.List(pl.String)
    )
    assert_series_equal(result["ngrams"], expected)


def test_custom_delimiter():
    df = pl.DataFrame({"words": [["foo", "bar", "baz"]]})
    result = df.with_columns(
        ngrams(pl.col("words"), n_range=[2], delimiter="_").alias("ngrams")
    )
    expected = pl.Series("ngrams", [["foo_bar", "bar_baz"]])
    assert_series_equal(result["ngrams"], expected)


def test_default_delimiter():
    df = pl.DataFrame({"words": [["hello", "world"]]})
    result = df.with_columns(ngrams(pl.col("words"), n_range=[2]).alias("ngrams"))
    expected = pl.Series("ngrams", [["hello world"]])
    assert_series_equal(result["ngrams"], expected)


def test_empty_list():
    df = pl.DataFrame({"words": [[]]})
    result = df.with_columns(ngrams(pl.col("words"), n_range=[2]).alias("ngrams"))
    expected = pl.Series("ngrams", [[]], dtype=pl.List(pl.String))
    assert_series_equal(result["ngrams"], expected)


def test_single_word():
    df = pl.DataFrame({"words": [["solo"]]})
    result = df.with_columns(ngrams(pl.col("words"), n_range=[2]).alias("ngrams"))
    expected = pl.Series("ngrams", [[]], dtype=pl.List(pl.String))
    assert_series_equal(result["ngrams"], expected)


def test_unigrams():
    df = pl.DataFrame({"words": [["alpha", "beta", "gamma"]]})
    result = df.with_columns(ngrams(pl.col("words"), n_range=[1]).alias("ngrams"))
    expected = pl.Series("ngrams", [["alpha", "beta", "gamma"]])
    assert_series_equal(result["ngrams"], expected)


def test_multiple_rows():
    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "words": [
                ["the", "quick", "brown"],
                ["lazy", "dog"],
                ["jumps", "over", "the", "fence"],
            ],
        }
    )
    result = df.with_columns(ngrams(pl.col("words"), n_range=[2]).alias("ngrams"))
    expected = pl.Series(
        "ngrams",
        [
            ["the quick", "quick brown"],
            ["lazy dog"],
            ["jumps over", "over the", "the fence"],
        ],
    )

    assert_series_equal(result["ngrams"], expected)


def test_with_lazy_api():
    df = pl.DataFrame({"words": [["hello", "beautiful", "world"]]})
    result = (
        df.lazy()
        .with_columns(ngrams(pl.col("words"), n_range=[2, 3]).alias("ngrams"))
        .collect()
    )
    expected = pl.Series(
        "ngrams", [["hello beautiful", "beautiful world", "hello beautiful world"]]
    )
    assert_series_equal(result["ngrams"], expected)


def test_with_filter():
    df = pl.DataFrame(
        {
            "category": ["A", "B", "A"],
            "words": [
                ["one", "two", "three"],
                ["four", "five"],
                ["six", "seven", "eight"],
            ],
        }
    )
    expected = pl.DataFrame(
        {
            "category": ["A", "A"],
            "words": [
                ["one", "two", "three"],
                ["six", "seven", "eight"],
            ],
            "ngrams": [
                ["one-two", "two-three"],
                ["six-seven", "seven-eight"],
            ],
        }
    )
    result = df.filter(pl.col("category") == "A").with_columns(
        ngrams(pl.col("words"), n_range=[2], delimiter="-").alias("ngrams")
    )
    assert_frame_equal(result, expected)


def test_large_n_range():
    df = pl.DataFrame({"words": [["a", "b"]]})
    result = df.with_columns(
        ngrams(pl.col("words"), n_range=[1, 2, 5, 10]).alias("ngrams")
    )
    expected = pl.Series("ngrams", [["a", "b", "a b"]])
    assert_series_equal(result["ngrams"], expected)


def test_groupby_aggregation():
    df = pl.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "words": [
                ["hello", "world"],
                ["foo", "bar"],
                ["alpha", "beta", "gamma"],
                ["one", "two"],
            ],
        }
    )
    expected = pl.DataFrame(
        {
            "group": ["A", "B"],
            "ngrams": [
                ["hello world", "foo bar"],
                ["alpha beta", "beta gamma", "one two"],
            ],
        }
    ).sort(by=["group", "ngrams"])
    result = (
        df.with_columns(ngrams(pl.col("words"), n_range=[2]).alias("ngrams"))
        .group_by("group")
        .agg(pl.col("ngrams").flatten())
        .sort(by=["group", "ngrams"])
    )
    assert_frame_equal(result, expected)
