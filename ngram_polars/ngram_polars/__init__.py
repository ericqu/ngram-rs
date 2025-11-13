from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function
from polars._typing import IntoExpr

PLUGIN_PATH = Path(__file__).parent

def ngrams(expr: IntoExpr, n_range: list[int] =[1], delimiter : str = " ") -> pl.Expr:
    """Return a list of n-grams given a list of strings"""
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="ngrams",
        args=[expr],
        kwargs={"n_range": n_range, "delimiter": delimiter},
        is_elementwise=True,
        changes_length=True,
    )
