//! # Short example illustrating the library usage
//!
//! ```
//! use std::borrow::Cow;
//! use ngram_rs::generate_ngrams;
//!
//! let words = vec!["the".to_string(), "quick".to_string(), "brown".to_string()];
//! let ngrams = generate_ngrams(&words, &[1, 2], None);
//!
//! assert_eq!(ngrams, vec![
//!     Cow::Borrowed("the"),
//!     Cow::Borrowed("quick"),
//!     Cow::Borrowed("brown"),
//!     Cow::Owned("the quick".to_string()),
//!     Cow::Owned("quick brown".to_string()),
//! ]);
//! ```

use std::borrow::Cow;

/// Generates n-grams from a sequence of words with configurable n-gram sizes and delimiter.
///
/// This function creates n-grams (contiguous sequences of n words) from the input words
/// for all specified n-values in the given range. It uses `Cow` (Copy on Write) to
/// avoid unnecessary allocations when possible.
///
/// # Arguments
///
/// * `words` - A slice of String objects representing the input text as individual words
/// * `n_range` - A slice of usize values specifying which n-gram sizes to generate
/// * `delimiter` - Optional delimiter string to use between words in n-grams (defaults to space)
///
/// # Returns
///
/// A vector of `Cow<str>` where:
/// - Unigrams (n=1) are returned as `Cow::Borrowed` to avoid allocation
/// - Bigrams and higher n-grams are returned as `Cow::Owned` strings
pub fn generate_ngrams<'a>(
    words: &'a [String],
    n_range: &[usize],
    delimiter: Option<&str>,
) -> Vec<Cow<'a, str>> {
    let delimiter = delimiter.unwrap_or(" ");
    let mut result = Vec::new();

    for &n in n_range {
        if n == 0 || n > words.len() {
            continue;
        }

        match n {
            1 => {
                // For unigrams, we can use references directly
                result.extend(words.iter().map(|w| Cow::Borrowed(w.as_str())));
            }
            2 => {
                // For bigrams, we can avoid some intermediate allocations
                for window in words.windows(2) {
                    let mut ngram =
                        String::with_capacity(window[0].len() + window[1].len() + delimiter.len());
                    ngram.push_str(&window[0]);
                    ngram.push_str(delimiter);
                    ngram.push_str(&window[1]);
                    result.push(Cow::Owned(ngram));
                }
            }
            _ => {
                // For higher n-grams, use the standard join
                for window in words.windows(n) {
                    let ngram = window.join(delimiter);
                    result.push(Cow::Owned(ngram));
                }
            }
        }
    }

    result
}

/// Generates n-grams and returns owned strings, useful for integration with Polars.
///
/// This is a convenience wrapper around `generate_ngrams` that converts all results
/// to owned strings. This is particularly useful when working with data frames or
/// other structures that require owned data.
///
/// # Arguments
///
/// * `words` - A slice of String objects representing the input text as individual words
/// * `n_range` - A slice of usize values specifying which n-gram sizes to generate
/// * `delimiter` - Delimiter string to use between words in n-grams
///
/// # Returns
///
/// A vector of owned String objects containing all generated n-grams
///
/// # Examples
///
/// ```
/// use ngram_rs::generate_ngrams_owned;
///
/// let words = vec!["hello".to_string(), "world".to_string()];
/// let ngrams = generate_ngrams_owned(&words, &[2], "-");
///
/// assert_eq!(ngrams, vec!["hello-world".to_string()]);
/// ```
pub fn generate_ngrams_owned(words: &[String], n_range: &[usize], delimiter: &str) -> Vec<String> {
    generate_ngrams(words, n_range, Some(delimiter))
        .into_iter()
        .map(|cow| cow.into_owned())
        .collect()
}

/// An iterator that generates n-grams lazily for memory-efficient processing.
///
/// This iterator produces n-grams on-demand rather than generating all at once,
/// which can be more memory efficient for large inputs or when only a subset of
/// n-grams is needed.
///
/// # Fields
///
/// * `words` - Reference to the input words slice
/// * `n_range` - Reference to the n-gram sizes to generate
/// * `current_n` - Current index in the n_range being processed
/// * `current_window` - Current starting position for the sliding window
/// * `delimiter` - Delimiter to use between words
pub struct NGramIterator<'a> {
    words: &'a [String],
    n_range: &'a [usize],
    current_n: usize,
    current_window: usize,
    delimiter: &'a str,
}

impl<'a> Iterator for NGramIterator<'a> {
    type Item = Cow<'a, str>;

    /// Returns the next n-gram in the sequence, or None when all n-grams have been generated.
    ///
    /// This implementation uses a state machine that:
    /// 1. Iterates through each n-value in n_range
    /// 2. For each n-value, slides a window through the words
    /// 3. Returns borrowed strings for unigrams, owned strings for higher n-grams
    fn next(&mut self) -> Option<Self::Item> {
        while self.current_n < self.n_range.len() {
            let n = self.n_range[self.current_n];

            // Skip invalid n-values
            if n == 0 || n > self.words.len() {
                self.current_n += 1;
                self.current_window = 0;
                continue;
            }

            // Check if we have more windows to process for current n-value
            if self.current_window + n <= self.words.len() {
                let window = &self.words[self.current_window..self.current_window + n];
                self.current_window += 1;

                return if n == 1 {
                    Some(Cow::Borrowed(window[0].as_str()))
                } else {
                    Some(Cow::Owned(window.join(self.delimiter)))
                };
            } else {
                self.current_n += 1;
                self.current_window = 0;
            }
        }

        None
    }
}

/// Creates an iterator that generates n-grams lazily.
///
/// This function is useful when you want to process n-grams one at a time
/// rather than generating all of them upfront, which can save memory for
/// large inputs.
///
/// # Arguments
///
/// * `words` - A slice of String objects representing the input text
/// * `n_range` - A slice of usize values specifying n-gram sizes
/// * `delimiter` - Optional delimiter string (defaults to space)
///
/// # Returns
///
/// An `NGramIterator` that yields n-grams as `Cow<str>` values
///
/// # Examples
///
/// ```
/// use std::borrow::Cow;
/// use ngram_rs::ngrams_as_iterator;
///
/// let words = vec!["a".to_string(), "b".to_string(), "c".to_string()];
/// let mut iter = ngrams_as_iterator(&words, &[2], Some("-"));
///
/// assert_eq!(iter.next(), Some(Cow::Owned("a-b".to_string())));
/// assert_eq!(iter.next(), Some(Cow::Owned("b-c".to_string())));
/// assert_eq!(iter.next(), None);
/// ```
pub fn ngrams_as_iterator<'a>(
    words: &'a [String],
    n_range: &'a [usize],
    delimiter: Option<&'a str>,
) -> NGramIterator<'a> {
    NGramIterator {
        words,
        n_range,
        current_n: 0,
        current_window: 0,
        delimiter: delimiter.unwrap_or(" "),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests basic n-gram generation with multiple n-values
    #[test]
    fn test_basic_ngrams() {
        let words = vec![
            "the".to_string(),
            "quick".to_string(),
            "brown".to_string(),
            "fox".to_string(),
        ];

        let result = generate_ngrams(&words, &[2, 3], None);
        assert_eq!(
            result,
            vec![
                Cow::<str>::Owned("the quick".to_string()),
                Cow::Owned("quick brown".to_string()),
                Cow::Owned("brown fox".to_string()),
                Cow::Owned("the quick brown".to_string()),
                Cow::Owned("quick brown fox".to_string()),
            ]
        );
    }

    /// Tests n-gram generation with custom delimiter
    #[test]
    fn test_custom_delimiter() {
        let words = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let result = generate_ngrams(&words, &[2], Some("-"));
        assert_eq!(
            result,
            vec![
                Cow::<str>::Owned("a-b".to_string()),
                Cow::Owned("b-c".to_string()),
            ]
        );
    }

    /// Tests n-gram generation with mixed n-values including unigrams
    #[test]
    fn test_mixed_n_range() {
        let words = vec!["x".to_string(), "y".to_string(), "z".to_string()];

        let result = generate_ngrams(&words, &[1, 3], None);
        assert_eq!(
            result,
            vec![
                Cow::Borrowed("x"),
                Cow::Borrowed("y"),
                Cow::Borrowed("z"),
                Cow::Owned("x y z".to_string()),
            ]
        );
    }

    /// Tests the iterator-based implementation
    #[test]
    fn test_ngram_iterator() {
        let words = vec!["1".to_string(), "2".to_string(), "3".to_string()];
        let mut iter = ngrams_as_iterator(&words, &[1, 2], None);

        assert_eq!(iter.next(), Some(Cow::Borrowed("1")));
        assert_eq!(iter.next(), Some(Cow::Borrowed("2")));
        assert_eq!(iter.next(), Some(Cow::Borrowed("3")));
        assert_eq!(iter.next(), Some(Cow::Owned("1 2".to_string())));
        assert_eq!(iter.next(), Some(Cow::Owned("2 3".to_string())));
        assert_eq!(iter.next(), None);
    }

    /// Tests the owned strings version
    #[test]
    fn test_owned_version() {
        let words = vec!["alpha".to_string(), "beta".to_string()];
        let result = generate_ngrams_owned(&words, &[2], "+");

        assert_eq!(result, vec!["alpha+beta".to_string()]);
    }
}
