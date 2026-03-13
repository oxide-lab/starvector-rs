use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MetadataError {
    #[error("failed to read file {path}: {source}")]
    ReadFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse json file {path}: {source}")]
    ParseJson {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("missing metadata file {path}")]
    MissingFile { path: PathBuf },
}

pub fn read_json_file<T: for<'de> Deserialize<'de>>(
    model_dir: &Path,
    file_name: &str,
) -> Result<T, MetadataError> {
    let path = model_dir.join(file_name);
    if !path.exists() {
        return Err(MetadataError::MissingFile { path });
    }

    let raw = fs::read_to_string(&path).map_err(|source| MetadataError::ReadFile {
        path: path.clone(),
        source,
    })?;
    serde_json::from_str(&raw).map_err(|source| MetadataError::ParseJson { path, source })
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub multi_query: bool,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub image_size: usize,
    pub image_encoder_type: String,
    #[serde(default)]
    pub hidden_size_scale: Option<usize>,
    #[serde(default)]
    pub use_cache: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProcessorConfig {
    pub mean: [f64; 3],
    pub std: [f64; 3],
    pub size: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PreprocessorConfig {
    pub mean: [f64; 3],
    #[serde(alias = "std", alias = "image_std")]
    pub image_std: [f64; 3],
    pub size: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TokenizerConfig {
    pub vocab_size: usize,
    pub pad_token: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AddedTokenDescriptor {
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpecialTokenDescriptor {
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpecialTokensMap {
    pub pad_token: SpecialTokenDescriptor,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelIndexMetadata {
    pub total_size: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelSafetensorsIndex {
    pub metadata: ModelIndexMetadata,
    pub weight_map: HashMap<String, String>,
}
