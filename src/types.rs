use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::config::{
    AddedTokenDescriptor, MetadataError, ModelConfig, ModelSafetensorsIndex, PreprocessorConfig,
    ProcessorConfig, SpecialTokenDescriptor, SpecialTokensMap, TokenizerConfig, read_json_file,
};

pub const MODEL_CONFIG_FILE: &str = "config.json";
pub const PROCESSOR_CONFIG_FILE: &str = "processor_config.json";
pub const PREPROCESSOR_CONFIG_FILE: &str = "preprocessor_config.json";
pub const TOKENIZER_JSON_FILE: &str = "tokenizer.json";
pub const TOKENIZER_CONFIG_FILE: &str = "tokenizer_config.json";
pub const SPECIAL_TOKENS_MAP_FILE: &str = "special_tokens_map.json";
pub const ADDED_TOKENS_FILE: &str = "added_tokens.json";
pub const MODEL_INDEX_FILE: &str = "model.safetensors.index.json";
pub const DECODER_EMBEDDING_KEY: &str = "model.svg_transformer.transformer.transformer.wte.weight";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecialTokenIds {
    pub pad: u32,
    pub svg_start: u32,
    pub image_start: u32,
    pub caption_start: u32,
}

#[derive(Debug, Clone)]
pub struct ParsedModelMetadata {
    pub model_config: ModelConfig,
    pub processor_config: ProcessorConfig,
    pub preprocessor_config: PreprocessorConfig,
    pub tokenizer_json: serde_json::Value,
    pub tokenizer_config: TokenizerConfig,
    pub special_tokens_map: SpecialTokensMap,
    pub added_tokens: HashMap<String, u32>,
    pub model_index: ModelSafetensorsIndex,
}

impl ParsedModelMetadata {
    pub fn from_model_dir(model_dir: &Path) -> Result<Self, MetadataError> {
        let model_config = read_json_file::<ModelConfig>(model_dir, MODEL_CONFIG_FILE)?;
        let processor_config = read_json_file::<ProcessorConfig>(model_dir, PROCESSOR_CONFIG_FILE)?;
        let preprocessor_config =
            read_json_file::<PreprocessorConfig>(model_dir, PREPROCESSOR_CONFIG_FILE)?;
        let tokenizer_json = read_json_file::<serde_json::Value>(model_dir, TOKENIZER_JSON_FILE)?;
        let tokenizer_config = read_json_file::<TokenizerConfig>(model_dir, TOKENIZER_CONFIG_FILE)?;
        let special_tokens_map =
            read_json_file::<SpecialTokensMap>(model_dir, SPECIAL_TOKENS_MAP_FILE)?;
        let added_tokens = read_json_file::<HashMap<String, u32>>(model_dir, ADDED_TOKENS_FILE)?;
        let model_index = read_json_file::<ModelSafetensorsIndex>(model_dir, MODEL_INDEX_FILE)?;

        Ok(Self {
            model_config,
            processor_config,
            preprocessor_config,
            tokenizer_json,
            tokenizer_config,
            special_tokens_map,
            added_tokens,
            model_index,
        })
    }

    pub fn resolved_vocab_size(&self) -> usize {
        self.model_config.vocab_size
    }

    pub fn tokenizer_plus_added_vocab_size(&self) -> usize {
        self.tokenizer_config.vocab_size + self.added_tokens.len()
    }

    pub fn required_files() -> &'static [&'static str] {
        &[
            MODEL_CONFIG_FILE,
            PROCESSOR_CONFIG_FILE,
            PREPROCESSOR_CONFIG_FILE,
            TOKENIZER_JSON_FILE,
            TOKENIZER_CONFIG_FILE,
            SPECIAL_TOKENS_MAP_FILE,
            ADDED_TOKENS_FILE,
            MODEL_INDEX_FILE,
        ]
    }

    pub fn special_token_ids(&self) -> SpecialTokenIds {
        SpecialTokenIds {
            pad: *self
                .added_tokens
                .get("[PAD]")
                .expect("missing [PAD] in added_tokens.json"),
            svg_start: *self
                .added_tokens
                .get("<svg-start>")
                .expect("missing <svg-start> in added_tokens.json"),
            image_start: *self
                .added_tokens
                .get("<image-start>")
                .expect("missing <image-start> in added_tokens.json"),
            caption_start: *self
                .added_tokens
                .get("<caption-start>")
                .expect("missing <caption-start> in added_tokens.json"),
        }
    }

    pub fn referenced_weight_files(&self) -> Vec<String> {
        let mut files: HashSet<String> = self.model_index.weight_map.values().cloned().collect();
        let mut files: Vec<String> = files.drain().collect();
        files.sort_unstable();
        files
    }
}

pub fn normalize_added_token_map(
    value: &HashMap<String, AddedTokenDescriptor>,
) -> HashMap<String, u32> {
    value
        .iter()
        .filter_map(|(id, token)| {
            id.parse::<u32>()
                .ok()
                .map(|parsed_id| (token.content.clone(), parsed_id))
        })
        .collect()
}

pub fn extract_pad_from_special_tokens(value: &SpecialTokensMap) -> &SpecialTokenDescriptor {
    &value.pad_token
}
