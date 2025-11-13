use std::borrow::Cow;

/// More efficient version that avoids some allocations for common cases
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

/// Generate n-grams and return owned strings (useful for Polars)
pub fn generate_ngrams_owned(words: &[String], n_range: &[usize], delimiter: &str) -> Vec<String> {
    generate_ngrams(words, n_range, Some(delimiter))
        .into_iter()
        .map(|cow| cow.into_owned())
        .collect()
}

/// Iterator-based version for maximum flexibility
pub struct NGramIterator<'a> {
    words: &'a [String],
    n_range: &'a [usize],
    current_n: usize,
    current_window: usize,
    delimiter: &'a str,
}

impl<'a> Iterator for NGramIterator<'a> {
    type Item = Cow<'a, str>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_n < self.n_range.len() {
            let n = self.n_range[self.current_n];

            if n == 0 || n > self.words.len() {
                self.current_n += 1;
                self.current_window = 0;
                continue;
            }

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
}
