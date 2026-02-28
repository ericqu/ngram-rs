use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct NGramsKwargs {
    n_range: Vec<usize>,
    #[serde(default = "default_delimiter")]
    delimiter: String,
}

fn default_delimiter() -> String {
    " ".to_string()
}

fn ngrams_impl(inputs: &[Series], kwargs: NGramsKwargs) -> PolarsResult<Series> {
    let series = &inputs[0];
    let ca = series.list()?;

    let out: ListChunked = ca.try_apply_amortized(|amort_series| {
        let series = amort_series.as_ref();

        if series.is_empty() {
            return Ok(StringChunked::from_iter(std::iter::empty::<String>()).into_series());
        }

        let words_ca = match series.str() {
            Ok(ca) => ca,
            Err(_) => {
                // If we can't get as string, return empty series
                return Ok(StringChunked::from_iter(std::iter::empty::<String>()).into_series());
            }
        };

        let words: Vec<String> = words_ca
            .into_iter()
            .flatten()
            .map(|s| s.to_string())
            .collect();

        if words.is_empty() {
            return Ok(StringChunked::from_iter(std::iter::empty::<String>()).into_series());
        }

        let ngrams = ngram_rs::generate_ngrams_owned(&words, &kwargs.n_range, &kwargs.delimiter);
        Ok(StringChunked::from_iter(ngrams).into_series())
    })?;

    Ok(out.into_series())
}

fn output_type_list_string(_input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "ngrams".into(),
        DataType::List(Box::new(DataType::String)),
    ))
}

#[polars_expr(output_type_func = output_type_list_string)]
fn ngrams(inputs: &[Series], kwargs: NGramsKwargs) -> PolarsResult<Series> {
    ngrams_impl(inputs, kwargs)
}
