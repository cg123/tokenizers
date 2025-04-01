#![allow(clippy::map_entry)]

use super::trainer::BpeTrainer;
use super::BPE;
use super::{Pair, WithFirstLastIterator, Word};
use crate::tokenizer::{AddedToken, Result, Trainer};
use crate::utils::progress::{ProgressBar, ProgressStyle};
use crate::{PreTokenizedString, PreTokenizer};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

// Helper struct for the priority queue
#[derive(Debug, Eq)]
struct Merge {
    pair: Pair,
    count: u64,
    pos: HashSet<usize>,
}

impl PartialEq for Merge {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}

impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Merge {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.count != other.count {
            self.count.cmp(&other.count)
        } else {
            // Here we want ascending order
            other.pair.cmp(&self.pair)
        }
    }
}

struct SuperBpeConfig {
    /// Configuration for the standard BpeTrainer
    standard_config: BpeTrainer,
    /// The transition point at which we switch from stage 1 (with whitespace pretokenization)
    /// to stage 2 (without whitespace pretokenization)
    transition_point: usize,
}

/// A `SuperBpeTrainerBuilder` can be used to create a `SuperBpeTrainer` with a custom
/// configuration.
pub struct SuperBpeTrainerBuilder {
    config: SuperBpeConfig,
}

impl Default for SuperBpeTrainerBuilder {
    fn default() -> Self {
        Self {
            config: SuperBpeConfig {
                standard_config: BpeTrainer::default(),
                transition_point: 10000, // Default to 10000 tokens for stage 1
            },
        }
    }
}

impl SuperBpeTrainerBuilder {
    /// Constructs a new `SuperBpeTrainerBuilder`
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the expected minimum frequency
    #[must_use]
    pub fn min_frequency(mut self, frequency: u64) -> Self {
        let mut config = self.config.standard_config.clone();
        config.min_frequency = frequency;
        self.config.standard_config = config;
        self
    }

    /// Set the vocabulary size
    #[must_use]
    pub fn vocab_size(mut self, size: usize) -> Self {
        let mut config = self.config.standard_config.clone();
        config.vocab_size = size;
        self.config.standard_config = config;
        self
    }

    /// Set whether to show progress
    #[must_use]
    pub fn show_progress(mut self, show: bool) -> Self {
        let mut config = self.config.standard_config.clone();
        config.show_progress = show;
        self.config.standard_config = config;
        self
    }

    /// Set the special tokens
    #[must_use]
    pub fn special_tokens(mut self, tokens: Vec<AddedToken>) -> Self {
        let mut config = self.config.standard_config.clone();
        config.special_tokens = tokens;
        self.config.standard_config = config;
        self
    }

    /// Set whether to limit the alphabet
    #[must_use]
    pub fn limit_alphabet(mut self, limit: usize) -> Self {
        let mut config = self.config.standard_config.clone();
        config.limit_alphabet = Some(limit);
        self.config.standard_config = config;
        self
    }

    /// Set the initial alphabet
    #[must_use]
    pub fn initial_alphabet(mut self, alphabet: HashSet<char>) -> Self {
        let mut config = self.config.standard_config.clone();
        config.initial_alphabet = alphabet;
        self.config.standard_config = config;
        self
    }

    /// Set the continuing_subword_prefix
    #[must_use]
    pub fn continuing_subword_prefix(mut self, prefix: String) -> Self {
        let mut config = self.config.standard_config.clone();
        config.continuing_subword_prefix = Some(prefix);
        self.config.standard_config = config;
        self
    }

    /// Set the end_of_word_suffix
    #[must_use]
    pub fn end_of_word_suffix(mut self, suffix: String) -> Self {
        let mut config = self.config.standard_config.clone();
        config.end_of_word_suffix = Some(suffix);
        self.config.standard_config = config;
        self
    }

    /// Set max_token_length
    #[must_use]
    pub fn max_token_length(mut self, max_token_length: Option<usize>) -> Self {
        let mut config = self.config.standard_config.clone();
        config.max_token_length = max_token_length;
        self.config.standard_config = config;
        self
    }

    /// Set the transition point (vocabulary size at which to switch from stage 1 to stage 2)
    #[must_use]
    pub fn transition_point(mut self, transition_point: usize) -> Self {
        self.config.transition_point = transition_point;
        self
    }

    /// Constructs the final SuperBpeTrainer
    pub fn build(self) -> SuperBpeTrainer {
        SuperBpeTrainer {
            standard_config: self.config.standard_config,
            transition_point: self.config.transition_point,
            words_stage1: HashMap::new(),
            words_stage2: HashMap::new(),
        }
    }
}

/// In charge of training a SuperBPE model
///
/// SuperBPE introduces a curriculum approach to BPE tokenization:
/// - Stage 1: Regular BPE training with whitespace pretokenization up to transition_point tokens
/// - Stage 2: Continue BPE training without whitespace pretokenization up to the final vocab_size
///
/// # Examples
///
/// ```
/// use tokenizers::tokenizer::Trainer;
/// use tokenizers::models::bpe::{BPE, SuperBpeTrainer};
///
/// let sequences = vec![ "Hello world" ];
/// let mut trainer = SuperBpeTrainer::default();
/// trainer.feed(sequences.iter(), |s| Ok(vec![s.to_owned()]));
///
/// let mut model = BPE::default();
/// let special_tokens = trainer.train(&mut model).unwrap();
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Eq)]
pub struct SuperBpeTrainer {
    /// The standard BpeTrainer configuration
    pub standard_config: BpeTrainer,
    /// The transition point at which we switch from stage 1 to stage 2
    pub transition_point: usize,

    // Internal storage of words for both stages
    words_stage1: HashMap<String, u64>, // With whitespace pretokenization
    words_stage2: HashMap<String, u64>, // Without whitespace pretokenization
}

impl Default for SuperBpeTrainer {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl SuperBpeTrainer {
    pub fn new(min_frequency: u64, vocab_size: usize, transition_point: usize) -> Self {
        Self {
            standard_config: BpeTrainer::new(min_frequency, vocab_size),
            transition_point,
            words_stage1: HashMap::new(),
            words_stage2: HashMap::new(),
        }
    }

    pub fn builder() -> SuperBpeTrainerBuilder {
        SuperBpeTrainerBuilder::new()
    }

    // Helper functions for progress updates
    fn setup_progress(&self) -> Option<ProgressBar> {
        if self.standard_config.show_progress {
            let p = ProgressBar::new(0);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {msg:<30!} {wide_bar} {pos:<9!}/{len:>9!}")
                    .expect("Invalid progress template"),
            );
            Some(p)
        } else {
            None
        }
    }

    fn update_progress(&self, p: &Option<ProgressBar>, len: usize, message: &str) {
        if let Some(p) = p {
            p.set_message(message.to_string());
            p.set_length(len as u64);
            p.reset();
        }
    }

    fn finalize_progress(&self, p: &Option<ProgressBar>, final_len: usize) {
        if let Some(p) = p {
            p.set_length(final_len as u64);
            p.finish();
            println!();
        }
    }

    /// Train a SuperBPE model in two stages
    pub fn do_train(&self, model: &mut BPE) -> Result<Vec<AddedToken>> {
        let progress = self.setup_progress();

        // Stage 1: Regular BPE with whitespace pretokenization
        if let Some(p) = &progress {
            p.set_message("Stage 1: BPE with whitespace pretokenization");
        }

        // Set the vocabulary size to the transition point for stage 1
        let mut stage1_trainer = self.standard_config.clone();
        stage1_trainer.vocab_size = self.transition_point;
        stage1_trainer.train(model)?;

        if let Some(p) = &progress {
            p.set_message("Stage 1 complete");
        }

        // If we've already reached our target vocab size, we're done
        if model.get_vocab().len() >= self.standard_config.vocab_size {
            if let Some(p) = &progress {
                p.finish();
            }
            return Ok(self.standard_config.special_tokens.clone());
        }

        // Stage 2: Continue BPE training without whitespace pretokenization
        if let Some(p) = &progress {
            p.set_message("Stage 2: BPE without whitespace pretokenization");
        }

        // Now we'll continue training with stage 2 words (without whitespace pretokenization)
        // but using the vocabulary we've built so far

        // Initialize our vocabulary maps from the current model's vocabulary
        let mut word_to_id = model.vocab.clone();
        let mut id_to_word: Vec<String> = vec!["".to_string(); word_to_id.len()];
        for (word, id) in &word_to_id {
            id_to_word[*id as usize] = word.clone();
        }

        let max_token_length = self.standard_config.max_token_length.unwrap_or(usize::MAX);

        // Tokenize the stage 2 words with the current vocabulary
        self.update_progress(&progress, self.words_stage2.len(), "Tokenize stage 2 words");
        let (mut words, counts) = {
            let mut words = Vec::with_capacity(self.words_stage2.len());
            let mut counts = Vec::with_capacity(self.words_stage2.len());

            for (word, count) in &self.words_stage2 {
                let mut current_word = Word::new();
                counts.push(*count);

                for (is_first, is_last, c) in word.chars().with_first_and_last() {
                    let mut s = c.to_string();
                    if word_to_id.contains_key(&s) {
                        // Found the initial char in the authorized alphabet

                        // Add the `continuing_subword_prefix` if relevant
                        if !is_first {
                            if let Some(prefix) = &self.standard_config.continuing_subword_prefix {
                                s = format!("{prefix}{s}");
                            }
                        }
                        // Add the `end_of_word_suffix` if relevant
                        if is_last {
                            if let Some(suffix) = &self.standard_config.end_of_word_suffix {
                                s = format!("{s}{suffix}");
                            }
                        }

                        // Insert the new formed string if necessary
                        if !word_to_id.contains_key(&s) {
                            id_to_word.push(s.clone());
                            word_to_id.insert(s.clone(), (id_to_word.len() - 1) as u32);
                        }
                        current_word.add(word_to_id[&s], 1); // We do not care about the len here
                    }
                }
                words.push(current_word);

                if let Some(p) = &progress {
                    p.inc(1);
                }
            }

            (words, counts)
        };
        self.finalize_progress(&progress, words.len());

        // Count pairs in words
        self.update_progress(&progress, words.len(), "Count pairs in stage 2");

        // Custom implementation of count_pairs
        let (mut pair_counts, mut where_to_update): (
            HashMap<Pair, i32>,
            HashMap<Pair, HashSet<usize>>,
        ) = words
            .iter()
            .enumerate()
            .map(|(i, word)| {
                let mut pair_counts = HashMap::new();
                let mut where_to_update: HashMap<Pair, HashSet<usize>> = HashMap::new();

                for window in word.get_chars().windows(2) {
                    let cur_pair: Pair = (window[0], window[1]);

                    // Initialize pair_counts and where_to_update for this pair if we just saw it
                    if !pair_counts.contains_key(&cur_pair) {
                        pair_counts.insert(cur_pair, 0);
                    }

                    // Then update counts
                    let count = counts[i];
                    where_to_update
                        .entry(cur_pair)
                        .and_modify(|h| {
                            h.insert(i);
                        })
                        .or_insert_with(|| {
                            let mut h = HashSet::new();
                            h.insert(i);
                            h
                        });
                    *pair_counts.get_mut(&cur_pair).unwrap() += count as i32;
                }

                if let Some(p) = &progress {
                    p.inc(1);
                }

                (pair_counts, where_to_update)
            })
            .fold(
                (HashMap::new(), HashMap::new()),
                |(mut pair_counts, mut where_to_update), (pc, wtu)| {
                    for (k, v) in pc {
                        pair_counts.entry(k).and_modify(|c| *c += v).or_insert(v);
                    }
                    for (k, v) in wtu {
                        where_to_update
                            .entry(k)
                            .and_modify(|set| *set = set.union(&v).copied().collect())
                            .or_insert(v);
                    }
                    (pair_counts, where_to_update)
                },
            );

        // Insert them in the queue
        let mut queue = BinaryHeap::with_capacity(pair_counts.len());
        where_to_update.drain().for_each(|(pair, pos)| {
            let count = pair_counts[&pair];
            if count > 0 {
                queue.push(Merge {
                    pair,
                    count: count as u64,
                    pos,
                });
            }
        });
        self.finalize_progress(&progress, words.len());

        // Do merges
        self.update_progress(
            &progress,
            self.standard_config.vocab_size,
            "Compute stage 2 merges",
        );
        let mut merges: Vec<(Pair, u32)> = vec![];

        // Get the initial number of tokens
        let initial_token_count = word_to_id.len();
        if let Some(p) = &progress {
            p.set_message(format!(
                "Starting stage 2 with {} tokens",
                initial_token_count
            ));
        }

        // Continue merging until we reach our target vocabulary size
        loop {
            // Stop if we've reached our vocabulary size target
            if word_to_id.len() >= self.standard_config.vocab_size {
                break;
            }

            if queue.is_empty() {
                break;
            }

            let mut top = queue.pop().unwrap();
            if top.count != pair_counts[&top.pair] as u64 {
                top.count = pair_counts[&top.pair] as u64;
                queue.push(top);
                continue;
            }

            if top.count < 1 || self.standard_config.min_frequency > top.count {
                break;
            }

            let part_a = &id_to_word[top.pair.0 as usize];
            let mut part_b = id_to_word[top.pair.1 as usize].to_owned();

            // Build new token
            if let Some(prefix) = &self.standard_config.continuing_subword_prefix {
                if part_b.starts_with(prefix) {
                    let prefix_byte_len = prefix.chars().map(|c| c.len_utf8()).sum();
                    part_b = part_b[prefix_byte_len..].to_string();
                }
            }
            let new_token = format!("{part_a}{part_b}");

            // Insert new token if it doesn't already exist
            let new_token_id = word_to_id
                .get(&new_token)
                .copied()
                .unwrap_or(id_to_word.len() as u32);
            if !word_to_id.contains_key(&new_token) {
                id_to_word.push(new_token.clone());
                word_to_id.insert(new_token.clone(), new_token_id);
            }
            merges.push((top.pair, new_token_id));

            // Merge the new pair in every word
            let pos: &HashSet<usize> = &top.pos;

            let words_len = words.len();
            struct WordPtr(*mut Word);
            unsafe impl Sync for WordPtr {}
            let word_start = WordPtr(words.as_mut_ptr());

            let changes = pos
                .iter()
                .flat_map(|&i| unsafe {
                    assert!(i < words_len);
                    let word = word_start.0.add(i);
                    (*word)
                        .merge(top.pair.0, top.pair.1, new_token_id, max_token_length)
                        .into_iter()
                        .map(|c| (c, i))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            // Update pair counts
            for ((pair, change), iw) in changes {
                let count = change * counts[iw] as i32;
                pair_counts
                    .entry(pair)
                    .and_modify(|c| *c += count)
                    .or_insert(count);
                if change > 0 {
                    where_to_update
                        .entry(pair)
                        .and_modify(|h| {
                            h.insert(iw);
                        })
                        .or_insert_with(|| {
                            let mut h = HashSet::new();
                            h.insert(iw);
                            h
                        });
                }
            }
            where_to_update.drain().for_each(|(pair, pos)| {
                let count = pair_counts[&pair];
                if count > 0 {
                    queue.push(Merge {
                        pair,
                        count: count as u64,
                        pos,
                    });
                }
            });

            if let Some(p) = &progress {
                p.inc(1);
            }
        }
        self.finalize_progress(&progress, merges.len());

        if let Some(p) = &progress {
            p.set_message(format!(
                "Stage 2 added {} new tokens",
                word_to_id.len() - initial_token_count
            ));
        }

        // Transfer vocabulary and merges to the model
        // This will combine both stage 1 and stage 2 vocabularies
        model.vocab = word_to_id;
        model.vocab_r = model
            .vocab
            .iter()
            .map(|(key, val)| (*val, key.to_owned()))
            .collect();

        // Add the new merges to the model
        for (i, (pair, new_token_id)) in merges.into_iter().enumerate() {
            if !model.merges.contains_key(&pair) {
                model
                    .merges
                    .insert(pair, (model.merges.len() as u32 + i as u32, new_token_id));
            }
        }

        // Update model configuration
        if let Some(prefix) = &self.standard_config.continuing_subword_prefix {
            model.continuing_subword_prefix = Some(prefix.to_owned());
        }
        if let Some(suffix) = &self.standard_config.end_of_word_suffix {
            model.end_of_word_suffix = Some(suffix.to_owned());
        }

        if let Some(p) = &progress {
            p.set_message("Stage 2 complete");
            p.finish();
        }

        Ok(self.standard_config.special_tokens.clone())
    }
}

impl Trainer for SuperBpeTrainer {
    type Model = BPE;

    /// Train a SuperBPE model
    fn train(&self, model: &mut BPE) -> Result<Vec<AddedToken>> {
        self.do_train(model)
    }

    /// Whether we should show progress
    fn should_show_progress(&self) -> bool {
        self.standard_config.show_progress
    }

    /// Feed the Trainer with a new corpus
    ///
    /// For SuperBPE, we need to process the corpus in two different ways:
    /// 1. With whitespace pretokenization (for stage 1)
    /// 2. Without whitespace pretokenization (for stage 2)
    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        // Process each sequence into two separate word maps
        let mut words_stage1 = HashMap::new();
        let mut words_stage2 = HashMap::new();

        let ws_pretok = crate::pre_tokenizers::split::Split::new(
            crate::pre_tokenizers::split::SplitPattern::Regex(r"\w+".to_string()),
            crate::SplitDelimiterBehavior::MergedWithPrevious,
            false,
        )?;

        let mut vector = Vec::new();
        // Process the iterator sequentially
        for sequence in iterator {
            vector.push(sequence.as_ref().to_string());
            for token in process(sequence.as_ref())? {
                words_stage2
                    .entry(token.clone())
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
            }
            let mut pretokenized: PreTokenizedString = sequence.as_ref().into();
            ws_pretok.pre_tokenize(&mut pretokenized)?;
            for (word, _, _) in
                pretokenized.get_splits(crate::OffsetReferential::Original, crate::OffsetType::Byte)
            {
                for token in process(word)? {
                    words_stage1
                        .entry(token.clone())
                        .and_modify(|count| *count += 1)
                        .or_insert(1);
                }
            }
        }

        self.words_stage1 = words_stage1;
        self.words_stage2 = words_stage2;

        let mut trainer = BpeTrainer::default();
        trainer.words = self.words_stage1.clone();
        trainer.vocab_size = self.transition_point;
        trainer.min_frequency = self.standard_config.min_frequency;
        trainer.show_progress = self.standard_config.show_progress;
        trainer.special_tokens = self.standard_config.special_tokens.clone();
        trainer.limit_alphabet = self.standard_config.limit_alphabet;
        trainer.initial_alphabet = self.standard_config.initial_alphabet.clone();
        trainer.continuing_subword_prefix = self.standard_config.continuing_subword_prefix.clone();
        trainer.end_of_word_suffix = self.standard_config.end_of_word_suffix.clone();
        trainer.max_token_length = self.standard_config.max_token_length;
        self.standard_config = trainer;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Decoder, NormalizedString, Normalizer, OffsetReferential, OffsetType, PreTokenizedString,
        Trainer,
    };

    use super::{SuperBpeTrainer, BPE};

    #[test]
    fn test_super_bpe_train() {
        let corpus = vec![
            "roses are red",
            "violets are blue",
            "BERT is big",
            "and GPT-2 is too",
            " ~~ ",
            "I love tokenizers",
            "honk honk",
            "here\t\tare some tab\tcharacters",
            "and some newlines\n\nbetween things",
        ];

        // Create a trainer with a small transition point to ensure both stages run
        let mut trainer = SuperBpeTrainer::builder()
            .show_progress(false)
            .min_frequency(0)
            .vocab_size(256)
            .transition_point(128)
            .build();

        let normalizer = crate::normalizers::ByteLevel::new();
        trainer
            .feed(corpus.iter(), |s: &str| {
                let mut normalized: NormalizedString = s.into();
                normalizer.normalize(&mut normalized)?;
                let pretokenized: PreTokenizedString = normalized.into();
                Ok(pretokenized
                    .get_splits(OffsetReferential::Original, OffsetType::Byte)
                    .into_iter()
                    .map(|(s, _, _)| s.to_owned())
                    .collect())
            })
            .unwrap();

        let mut model = BPE::default();

        // Train the model
        trainer.do_train(&mut model).unwrap();

        // Verify that the model has a vocabulary with both subword tokens and
        // some superword tokens (tokens that span word boundaries)
        let vocab = model.get_vocab();
        eprintln!("Vocabulary size: {}", vocab.len());

        let pp = crate::processors::byte_level::ByteLevel::new(false, false, false);
        let decoded_vocab = vocab
            .keys()
            .map(|token| pp.decode(vec![token.clone()]).unwrap())
            .collect::<Vec<_>>();
        // The vocab should contain some superword tokens like "are " or " is"
        assert!(decoded_vocab
            .iter()
            .any(|token| token.chars().skip(1).any(|c| c == ' ')));
    }
}
