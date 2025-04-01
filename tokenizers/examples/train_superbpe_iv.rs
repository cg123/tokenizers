use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use tokenizers::models::bpe::BPE;
use tokenizers::models::bpe::{BpeTrainer, BpeTrainerBuilder};
use tokenizers::normalizers::ByteLevel as ByteLevelNormalizer;
use tokenizers::pre_tokenizers::digits::Digits;
use tokenizers::pre_tokenizers::Sequence;
use tokenizers::processors::byte_level::ByteLevel as ByteLevelPostProcessor;
use tokenizers::{
    pre_tokenizers, AddedToken, DecoderWrapper, NormalizerWrapper, PostProcessorWrapper, Split,
    SplitDelimiterBehavior,
};
use tokenizers::{PreTokenizerWrapper, TokenizerImpl};

const VOCAB_SIZE: usize = 256_000;
const TRANSITION_POINT: usize = 180_000;

fn main() {
    let corpus = BufReader::new(
        File::open(Path::new("/workspace/fasttext-train-multilingual.txt")).unwrap(),
    )
    .lines()
    .map(|line| line.unwrap())
    .collect::<Vec<_>>();

    let mut trainer_0 = BpeTrainerBuilder::default()
        .vocab_size(TRANSITION_POINT)
        .min_frequency(2)
        .max_token_length(Some(256))
        .show_progress(true)
        .build();
    type Tok = TokenizerImpl<
        BPE,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >;
    let mut tokenizer_0 = Tok::new(BPE::default());
    let tokenizer_0 = tokenizer_0
        .with_normalizer(Some(
            NormalizerWrapper::from(ByteLevelNormalizer::default()),
        ))
        .with_pre_tokenizer(Some(PreTokenizerWrapper::Sequence(Sequence::new(vec![
            PreTokenizerWrapper::Digits(Digits::default()),
            PreTokenizerWrapper::Split(
                tokenizers::pre_tokenizers::split::Split::new(
                    pre_tokenizers::split::SplitPattern::Regex(r"\w+".to_string()),
                    SplitDelimiterBehavior::MergedWithPrevious,
                    false,
                )
                .unwrap(),
            ),
        ]))))
        .with_post_processor(Some(PostProcessorWrapper::from(
            ByteLevelPostProcessor::default(),
        )))
        .train(&mut trainer_0, corpus.iter())
        .unwrap();

    let vocab_0 = tokenizer_0
        .get_vocab(false)
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    println!("Trained subword tokenizer, vocab size: {}", vocab_0.len());
    let mut trainer_1 = BpeTrainerBuilder::default()
        .vocab_size(VOCAB_SIZE)
        .min_frequency(2)
        .max_token_length(Some(256))
        .show_progress(true)
        .initial_tokens(vocab_0)
        .build();
    let mut tokenizer_1 = Tok::new(BPE::default());
    let tokenizer_1 = tokenizer_1
        .with_normalizer(Some(
            NormalizerWrapper::from(ByteLevelNormalizer::default()),
        ))
        .with_pre_tokenizer(Some(PreTokenizerWrapper::Sequence(Sequence::new(vec![
            PreTokenizerWrapper::Digits(Digits::default()),
        ]))))
        .with_post_processor(Some(PostProcessorWrapper::from(
            ByteLevelPostProcessor::default(),
        )))
        .train(&mut trainer_1, corpus.iter())
        .unwrap();

    println!("Tokenizer trained successfully!");
    println!("Vocab size: {}", tokenizer_1.get_vocab_size(true));
    tokenizer_1
        .save("superbpe_tokenizer_out.json", true)
        .unwrap();
}
