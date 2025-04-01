use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use tokenizers::models::bpe::super_trainer::SuperBpeTrainerBuilder;
use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::{ByteLevel as ByteLevelNormalizer};
use tokenizers::pre_tokenizers::digits::Digits;
use tokenizers::processors::byte_level::ByteLevel as ByteLevelPostProcessor;
use tokenizers::{AddedToken, DecoderWrapper, NormalizerWrapper, PostProcessorWrapper};
use tokenizers::TokenizerImpl;

fn main() {
    let corpus = BufReader::new(
        File::open(Path::new("/workspace/fasttext-train-multilingual.txt")).unwrap(),
    )
    .lines()
    .map(|line| line.unwrap())
    .collect::<Vec<_>>();
    let mut trainer = SuperBpeTrainerBuilder::default()
        .vocab_size(256_000)
        .transition_point(180_000)
        .min_frequency(2)
        .special_tokens(vec![
            AddedToken::from("<|begin_of_text|>", true),
            AddedToken::from("<|end_of_text|>", true),
            AddedToken::from("<|pad|>", true),
        ])
        .show_progress(true)
        .build();

    type Tok = TokenizerImpl<BPE, NormalizerWrapper, Digits, PostProcessorWrapper, DecoderWrapper>;
    let mut tokenizer = Tok::new(BPE::default());
    let tokenizer = tokenizer
        .with_normalizer(Some(
            NormalizerWrapper::from(ByteLevelNormalizer::default()),
        ))
        .with_pre_tokenizer(Some(Digits::default()))
        .with_post_processor(Some(PostProcessorWrapper::from(
            ByteLevelPostProcessor::default(),
        )))
        .train(&mut trainer, corpus.iter())
        .unwrap();

    println!("Tokenizer trained successfully!");
    println!("Vocab size: {}", tokenizer.get_vocab_size(true));
    tokenizer.save("superbpe_tokenizer_out.json", true).unwrap();
}
