use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use itertools::Itertools;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::BpeTrainerBuilder;
use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel as ByteLevelPreTokenizer;
use tokenizers::pre_tokenizers::digits::Digits;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::processors::byte_level::ByteLevel as ByteLevelPostProcessor;
use tokenizers::{
    pre_tokenizers, AddedToken, DecoderWrapper, NormalizerWrapper, PostProcessorWrapper,
    SplitDelimiterBehavior,
};
use tokenizers::{PreTokenizerWrapper, TokenizerImpl};

const VOCAB_SIZE: usize = 256_000;
const TRANSITION_POINT: usize = 180_000;

fn get_corpus() -> impl Iterator<Item = String> {
    BufReader::new(File::open(Path::new("/workspace/fasttext-train-multilingual.txt")).unwrap())
        .lines()
        .map(|line| line.unwrap())
}

fn main() {
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
            NormalizerWrapper::from(NFC::default()),
        ))
        .with_pre_tokenizer(Some(PreTokenizerWrapper::Sequence(Sequence::new(vec![
            PreTokenizerWrapper::Digits(Digits::new(true)),
            PreTokenizerWrapper::Split(
                tokenizers::pre_tokenizers::split::Split::new(
                    pre_tokenizers::split::SplitPattern::Regex(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+".to_string()),
                    SplitDelimiterBehavior::Isolated,
                    false,
                )
                .unwrap(),
            ),
            PreTokenizerWrapper::ByteLevel(ByteLevelPreTokenizer::new(false, false, false)),
        ]))))
        .with_decoder(Some(ByteLevelDecoder::new(false, false, false)))
        .with_post_processor(Some(PostProcessorWrapper::from(
            ByteLevelPostProcessor::new(false, false, false))))
        .train(&mut trainer_0, get_corpus())
        .unwrap();

    let vocab_0 = tokenizer_0
        .get_vocab(false)
        .iter()
        .sorted_by(|a, b| a.1.cmp(b.1))
        .map(|(k, _)| k.to_string())
        .collect::<Vec<_>>();
    println!("Trained subword tokenizer, vocab size: {}", vocab_0.len());
    tokenizer_0
        .save("superbpe_tokenizer_32k_stage0_out.json", true)
        .unwrap();

    let mut trainer_1 = BpeTrainerBuilder::default()
        .vocab_size(VOCAB_SIZE)
        .min_frequency(2)
        .max_token_length(Some(256))
        .show_progress(true)
        .initial_tokens(vocab_0)
        .special_tokens(vec![
            AddedToken::from("<|begin_of_text|>", true),
            AddedToken::from("<|end_of_text|>", true),
            AddedToken::from("<|pad|>", true),
        ])
        .build();
    let mut tokenizer_1 = Tok::new(BPE::default());
    let tokenizer_1 = tokenizer_1
        .with_normalizer(Some(NormalizerWrapper::from(NFC::default())))
        .with_pre_tokenizer(Some(PreTokenizerWrapper::Sequence(Sequence::new(vec![
            PreTokenizerWrapper::Digits(Digits::default()),
            PreTokenizerWrapper::ByteLevel(ByteLevelPreTokenizer::new(false, false, false)),
        ]))))
        .with_post_processor(Some(PostProcessorWrapper::from(
            ByteLevelPostProcessor::new(false, false, false),
        )))
        .with_decoder(Some(ByteLevelDecoder::new(false, false, false)))
        .train(&mut trainer_1, get_corpus())
        .unwrap();

    println!("Tokenizer trained successfully!");
    println!("Vocab size: {}", tokenizer_1.get_vocab_size(true));
    tokenizer_1
        .save("superbpe_tokenizer_32k_out.json", true)
        .unwrap();
}
