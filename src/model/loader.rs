use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle::quantized::gguf_file;
use candle::{DType, Device};
use candle_nn::VarBuilder;
use safetensors::{SafeTensors, tensor::Dtype as SafeTensorDtype};
use thiserror::Error;

use crate::config::MetadataError;
use crate::types::ParsedModelMetadata;

const DECODER_PREFIX: &str = "model.svg_transformer.transformer";
const VISION_PREFIX: &str = "model.image_encoder";
const ADAPTER_PREFIX: &str = "model.image_projection";

#[derive(Debug, Clone, Copy)]
pub struct LoaderPrecisionPolicy {
    pub decoder_dtype: DType,
    pub vision_dtype: DType,
    pub adapter_dtype: DType,
}

impl LoaderPrecisionPolicy {
    pub const fn cpu_default() -> Self {
        Self {
            decoder_dtype: DType::F32,
            vision_dtype: DType::F32,
            adapter_dtype: DType::F32,
        }
    }

    pub const fn cuda_default() -> Self {
        Self {
            decoder_dtype: DType::F32,
            vision_dtype: DType::F16,
            adapter_dtype: DType::F16,
        }
    }

    pub fn for_device(device: &Device) -> Self {
        if device.is_cuda() {
            Self::cuda_default()
        } else {
            Self::cpu_default()
        }
    }
}

pub struct LoaderViews {
    pub decoder: VarBuilder<'static>,
    pub vision: VarBuilder<'static>,
    pub adapter: VarBuilder<'static>,
}

#[derive(Debug, Clone)]
pub struct NonFloatTensor {
    pub dtype: SafeTensorDtype,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl NonFloatTensor {
    pub fn as_i64_scalar(&self) -> Result<i64, LoaderError> {
        if self.dtype != SafeTensorDtype::I64 {
            return Err(LoaderError::UnexpectedNonFloatDtype {
                expected: SafeTensorDtype::I64,
                got: self.dtype,
            });
        }
        if !(self.shape.is_empty() || self.shape == [1]) {
            return Err(LoaderError::UnexpectedShape {
                name: "<non-float-scalar>".to_owned(),
                shape: self.shape.clone(),
            });
        }
        if self.data.len() != std::mem::size_of::<i64>() {
            return Err(LoaderError::InvalidDataLength {
                expected: std::mem::size_of::<i64>(),
                got: self.data.len(),
            });
        }
        let mut bytes = [0_u8; 8];
        bytes.copy_from_slice(&self.data);
        Ok(i64::from_le_bytes(bytes))
    }
}

#[derive(Debug, Error)]
pub enum LoaderError {
    #[error(transparent)]
    Metadata(#[from] MetadataError),
    #[error("missing shard file {path}")]
    MissingShard { path: PathBuf },
    #[error("tensor `{name}` missing from model.safetensors.index.json")]
    MissingTensorInIndex { name: String },
    #[error("failed to create mmap safetensors varbuilder: {0}")]
    VarBuilder(#[from] candle::Error),
    #[error("failed to read shard file {path}: {source}")]
    ReadShard {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse safetensors for {path}: {source}")]
    ParseSafetensors {
        path: PathBuf,
        #[source]
        source: safetensors::SafeTensorError,
    },
    #[error("failed to resolve tensor `{name}` in shard {path}: {source}")]
    ResolveTensor {
        name: String,
        path: PathBuf,
        #[source]
        source: safetensors::SafeTensorError,
    },
    #[error("tensor `{name}` is floating dtype {dtype:?}, use VarBuilder float path instead")]
    FloatTensorRequestedAsNonFloat {
        name: String,
        dtype: SafeTensorDtype,
    },
    #[error("unexpected tensor shape for `{name}`: {shape:?}")]
    UnexpectedShape { name: String, shape: Vec<usize> },
    #[error("unexpected non-float dtype, expected {expected:?}, got {got:?}")]
    UnexpectedNonFloatDtype {
        expected: SafeTensorDtype,
        got: SafeTensorDtype,
    },
    #[error("invalid tensor data length, expected {expected}, got {got}")]
    InvalidDataLength { expected: usize, got: usize },
    #[error("failed to read gguf file {path}: {source}")]
    ReadGguf {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse gguf content {path}: {source}")]
    ReadGgufContent {
        path: PathBuf,
        #[source]
        source: Box<candle::Error>,
    },
    #[error("failed to read gguf tensor `{name}` from {path}: {source}")]
    ReadGgufTensor {
        path: PathBuf,
        name: String,
        #[source]
        source: Box<candle::Error>,
    },
    #[error("non-float scalar `{name}` extraction from gguf is not supported")]
    NonFloatFromGgufNotSupported { name: String },
}

#[derive(Debug, Clone)]
pub struct ReuseFirstWeightLoader {
    model_dir: PathBuf,
    shard_paths: Vec<PathBuf>,
    weight_map: HashMap<String, String>,
    gguf_weights: Option<PathBuf>,
}

impl ReuseFirstWeightLoader {
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self, LoaderError> {
        Self::from_model_dir_with_gguf(model_dir, None::<&Path>)
    }

    pub fn from_model_dir_with_gguf(
        model_dir: impl AsRef<Path>,
        gguf_weights: Option<impl AsRef<Path>>,
    ) -> Result<Self, LoaderError> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let metadata = ParsedModelMetadata::from_model_dir(&model_dir)?;
        let shard_files = metadata.referenced_weight_files();

        let mut shard_paths = Vec::with_capacity(shard_files.len());
        for shard in &shard_files {
            let path = model_dir.join(shard);
            if !path.exists() {
                return Err(LoaderError::MissingShard { path });
            }
            shard_paths.push(path);
        }

        Ok(Self {
            model_dir,
            shard_paths,
            weight_map: metadata.model_index.weight_map,
            gguf_weights: gguf_weights.map(|p| p.as_ref().to_path_buf()),
        })
    }

    pub fn shard_paths(&self) -> &[PathBuf] {
        &self.shard_paths
    }

    pub fn make_views(
        &self,
        policy: LoaderPrecisionPolicy,
        device: &Device,
    ) -> Result<LoaderViews, LoaderError> {
        if let Some(gguf_path) = &self.gguf_weights {
            return self.make_views_from_gguf(gguf_path, policy, device, true);
        }

        // Reuse-first float loading path through Candle mmaped safetensors.
        let decoder_root = unsafe {
            VarBuilder::from_mmaped_safetensors(
                self.shard_paths.as_slice(),
                policy.decoder_dtype,
                device,
            )
        }?;
        let vision_root = unsafe {
            VarBuilder::from_mmaped_safetensors(
                self.shard_paths.as_slice(),
                policy.vision_dtype,
                device,
            )
        }?;
        let adapter_root = unsafe {
            VarBuilder::from_mmaped_safetensors(
                self.shard_paths.as_slice(),
                policy.adapter_dtype,
                device,
            )
        }?;

        Ok(LoaderViews {
            decoder: decoder_root.pp(DECODER_PREFIX),
            vision: vision_root.pp(VISION_PREFIX),
            adapter: adapter_root.pp(ADAPTER_PREFIX),
        })
    }

    pub fn make_views_for_gguf_runtime(
        &self,
        policy: LoaderPrecisionPolicy,
        device: &Device,
    ) -> Result<LoaderViews, LoaderError> {
        match &self.gguf_weights {
            Some(gguf_path) => self.make_views_from_gguf(gguf_path, policy, device, false),
            None => self.make_views(policy, device),
        }
    }

    fn make_views_from_gguf(
        &self,
        gguf_path: &Path,
        policy: LoaderPrecisionPolicy,
        device: &Device,
        include_decoder: bool,
    ) -> Result<LoaderViews, LoaderError> {
        let mut reader =
            std::io::BufReader::new(std::fs::File::open(gguf_path).map_err(|source| {
                LoaderError::ReadGguf {
                    path: gguf_path.to_path_buf(),
                    source,
                }
            })?);
        let content = gguf_file::Content::read(&mut reader).map_err(|source| {
            LoaderError::ReadGgufContent {
                path: gguf_path.to_path_buf(),
                source: Box::new(source),
            }
        })?;

        let mut tensors = HashMap::with_capacity(content.tensor_infos.len());
        for name in content.tensor_infos.keys() {
            if !include_decoder && name.starts_with(DECODER_PREFIX) {
                continue;
            }
            if content
                .tensor_infos
                .get(name)
                .is_some_and(|info| info.shape.rank() == 0)
            {
                continue;
            }
            let qtensor = content
                .tensor(&mut reader, name, device)
                .map_err(|source| LoaderError::ReadGgufTensor {
                    path: gguf_path.to_path_buf(),
                    name: name.clone(),
                    source: Box::new(source),
                })?;

            let tensor = match qtensor.dtype() {
                candle::quantized::GgmlDType::F32
                | candle::quantized::GgmlDType::F16
                | candle::quantized::GgmlDType::BF16 => qtensor.dequantize(device),
                _ if device.is_cuda() => qtensor.dequantize_f16(device),
                _ => qtensor.dequantize(device),
            }
            .map_err(|source| LoaderError::ReadGgufTensor {
                path: gguf_path.to_path_buf(),
                name: name.clone(),
                source: Box::new(source),
            })?;

            tensors.insert(name.clone(), tensor);
        }

        let root = VarBuilder::from_tensors(tensors, DType::F32, device);
        Ok(LoaderViews {
            decoder: root
                .clone()
                .set_dtype(policy.decoder_dtype)
                .pp(DECODER_PREFIX),
            vision: root
                .clone()
                .set_dtype(policy.vision_dtype)
                .pp(VISION_PREFIX),
            adapter: root.set_dtype(policy.adapter_dtype).pp(ADAPTER_PREFIX),
        })
    }

    pub fn load_non_float_tensor(&self, full_name: &str) -> Result<NonFloatTensor, LoaderError> {
        if self.gguf_weights.is_some() {
            return Err(LoaderError::NonFloatFromGgufNotSupported {
                name: full_name.to_owned(),
            });
        }
        let shard_rel =
            self.weight_map
                .get(full_name)
                .ok_or_else(|| LoaderError::MissingTensorInIndex {
                    name: full_name.to_owned(),
                })?;
        let shard_path = self.model_dir.join(shard_rel);
        let bytes = std::fs::read(&shard_path).map_err(|source| LoaderError::ReadShard {
            path: shard_path.clone(),
            source,
        })?;
        let safetensors =
            SafeTensors::deserialize(&bytes).map_err(|source| LoaderError::ParseSafetensors {
                path: shard_path.clone(),
                source,
            })?;
        let tensor =
            safetensors
                .tensor(full_name)
                .map_err(|source| LoaderError::ResolveTensor {
                    name: full_name.to_owned(),
                    path: shard_path.clone(),
                    source,
                })?;
        if is_float_dtype(tensor.dtype()) {
            return Err(LoaderError::FloatTensorRequestedAsNonFloat {
                name: full_name.to_owned(),
                dtype: tensor.dtype(),
            });
        }

        Ok(NonFloatTensor {
            dtype: tensor.dtype(),
            shape: tensor.shape().to_vec(),
            data: tensor.data().to_vec(),
        })
    }

    pub fn load_i64_scalar_non_float(&self, full_name: &str) -> Result<i64, LoaderError> {
        if self.gguf_weights.is_some() {
            return Ok(0);
        }
        let tensor = self.load_non_float_tensor(full_name)?;
        if !(tensor.shape.is_empty() || tensor.shape == [1]) {
            return Err(LoaderError::UnexpectedShape {
                name: full_name.to_owned(),
                shape: tensor.shape,
            });
        }
        tensor.as_i64_scalar()
    }
}

fn is_float_dtype(dtype: SafeTensorDtype) -> bool {
    matches!(
        dtype,
        SafeTensorDtype::BF16
            | SafeTensorDtype::F16
            | SafeTensorDtype::F32
            | SafeTensorDtype::F64
            | SafeTensorDtype::F8_E4M3
            | SafeTensorDtype::F8_E5M2
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const BN_TRACKED_KEY: &str = "model.image_projection.norm.num_batches_tracked";

    fn model_dir() -> PathBuf {
        if let Ok(path) = std::env::var("STARVECTOR_MODEL_DIR") {
            return PathBuf::from(path);
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("models")
            .join("starvector-1b-im2svg")
    }

    #[test]
    fn shard_discovery_is_deduplicated_and_existing() -> Result<(), LoaderError> {
        let loader = ReuseFirstWeightLoader::from_model_dir(model_dir())?;
        let shards = loader.shard_paths();
        assert_eq!(shards.len(), 2, "expected two safetensor shards");
        for shard in shards {
            assert!(shard.exists(), "missing shard file {}", shard.display());
        }
        Ok(())
    }

    #[test]
    fn float_loader_views_resolve_expected_prefixes() -> Result<(), LoaderError> {
        let loader = ReuseFirstWeightLoader::from_model_dir(model_dir())?;
        let views = loader.make_views(LoaderPrecisionPolicy::cpu_default(), &Device::Cpu)?;

        assert!(views.decoder.contains_tensor("transformer.wte.weight"));
        assert!(views.vision.contains_tensor("visual_encoder.conv1.weight"));
        assert!(views.adapter.contains_tensor("norm.weight"));

        let decoder_t = views.decoder.get_unchecked("transformer.wte.weight")?;
        let vision_t = views.vision.get_unchecked("visual_encoder.conv1.weight")?;
        let adapter_t = views.adapter.get_unchecked("c_fc.weight")?;

        assert_eq!(decoder_t.dtype(), DType::F32);
        assert_eq!(vision_t.dtype(), DType::F32);
        assert_eq!(adapter_t.dtype(), DType::F32);
        Ok(())
    }

    #[test]
    fn non_float_buffers_are_loaded_without_float_cast() -> Result<(), LoaderError> {
        let loader = ReuseFirstWeightLoader::from_model_dir(model_dir())?;
        let non_float = loader.load_non_float_tensor(BN_TRACKED_KEY)?;
        assert_eq!(non_float.dtype, SafeTensorDtype::I64);

        let value = loader.load_i64_scalar_non_float(BN_TRACKED_KEY)?;
        assert!(value >= 0);
        Ok(())
    }
}
