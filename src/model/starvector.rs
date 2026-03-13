use std::path::Path;

use candle::{DType, Device, Result, Tensor};
use tokenizers::Tokenizer;

use crate::types::{ParsedModelMetadata, TOKENIZER_JSON_FILE};

use super::adapter::StarVectorAdapter;
use super::adapter_quantized::StarVectorAdapterQuantized;
use super::bigcode::{BigCodeConfig, BigCodeDecoder};
use super::bigcode_quantized::BigCodeDecoderQuantized;
use super::generation::{GenerationConfig, GenerationOutput, has_suffix, next_token_id};
use super::image_preprocessor::ImagePreprocessor;
use super::loader::{LoaderPrecisionPolicy, ReuseFirstWeightLoader};
use super::vision::{StarVectorVisionConfig, StarVectorVisionTransformer};
use super::vision_quantized::StarVectorVisionTransformerQuantized;
use candle_transformers::generation::LogitsProcessor;

#[derive(Debug, Clone, Copy)]
pub struct PrecisionPolicy {
    pub decoder_dtype: DType,
    pub vision_dtype: DType,
    pub adapter_dtype: DType,
}

impl PrecisionPolicy {
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

pub struct StarVector {
    device: Device,
    precision: PrecisionPolicy,
    tokenizer: Tokenizer,
    prompt_tokens: Vec<u32>,
    stop_tokens: Vec<u32>,
    preprocessor: ImagePreprocessor,
    vision: VisionRuntime,
    adapter: AdapterRuntime,
    decoder: DecoderRuntime,
}

enum VisionRuntime {
    Float(StarVectorVisionTransformer),
    Quantized(StarVectorVisionTransformerQuantized),
}

impl VisionRuntime {
    fn forward_sequence(&self, pixel_values: &Tensor) -> Result<Tensor> {
        match self {
            Self::Float(v) => v.forward_sequence(pixel_values),
            Self::Quantized(v) => v.forward_sequence(pixel_values),
        }
    }
}

enum AdapterRuntime {
    Float(StarVectorAdapter),
    Quantized(StarVectorAdapterQuantized),
}

impl AdapterRuntime {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Float(a) => a.forward(xs),
            Self::Quantized(a) => a.forward(xs),
        }
    }
}

enum DecoderRuntime {
    Float(BigCodeDecoder),
    Quantized(BigCodeDecoderQuantized),
}

impl DecoderRuntime {
    fn config(&self) -> &BigCodeConfig {
        match self {
            Self::Float(d) => d.config(),
            Self::Quantized(d) => d.config(),
        }
    }

    fn clear_kv_cache(&mut self) {
        match self {
            Self::Float(d) => d.clear_kv_cache(),
            Self::Quantized(d) => d.clear_kv_cache(),
        }
    }

    fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Float(d) => d.embed_tokens(input_ids),
            Self::Quantized(d) => d.embed_tokens(input_ids),
        }
    }

    fn forward_inputs_embeds(&mut self, input_embeds: &Tensor, past_len: usize) -> Result<Tensor> {
        match self {
            Self::Float(d) => d.forward_inputs_embeds(input_embeds, past_len),
            Self::Quantized(d) => d.forward_inputs_embeds(input_embeds, past_len),
        }
    }

    fn forward(&mut self, input_ids: &Tensor, past_len: usize) -> Result<Tensor> {
        match self {
            Self::Float(d) => d.forward(input_ids, past_len),
            Self::Quantized(d) => d.forward(input_ids, past_len),
        }
    }
}

impl StarVector {
    pub fn load(
        model_dir: impl AsRef<Path>,
        device: &Device,
        precision: PrecisionPolicy,
    ) -> candle::Result<Self> {
        Self::load_with_gguf(model_dir, None::<&Path>, device, precision)
    }

    pub fn load_with_gguf(
        model_dir: impl AsRef<Path>,
        gguf_weights: Option<impl AsRef<Path>>,
        device: &Device,
        precision: PrecisionPolicy,
    ) -> candle::Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let gguf_path = gguf_weights.map(|p| p.as_ref().to_path_buf());
        let metadata = ParsedModelMetadata::from_model_dir(&model_dir)
            .map_err(|e| candle::Error::msg(e.to_string()))?;

        let preprocessor = ImagePreprocessor::from_model_dir(&model_dir)
            .map_err(|e| candle::Error::msg(e.to_string()))?;
        let vision_cfg = vision_config_from_metadata(&metadata);
        let decoder_cfg = BigCodeConfig::from_model_config(&metadata.model_config);
        let (vision, adapter, decoder) = if let Some(gguf_path) = gguf_path {
            let vision =
                VisionRuntime::Quantized(StarVectorVisionTransformerQuantized::load_from_gguf(
                    &gguf_path,
                    device,
                    &vision_cfg,
                )?);
            let adapter = AdapterRuntime::Quantized(StarVectorAdapterQuantized::load_from_gguf(
                &gguf_path, device,
            )?);
            let decoder = DecoderRuntime::Quantized(BigCodeDecoderQuantized::load_from_gguf(
                &gguf_path,
                device,
                decoder_cfg,
            )?);
            (vision, adapter, decoder)
        } else {
            let loader =
                ReuseFirstWeightLoader::from_model_dir_with_gguf(&model_dir, None::<&Path>)
                    .map_err(|e| candle::Error::msg(e.to_string()))?;
            let loader_views = loader
                .make_views(
                    LoaderPrecisionPolicy {
                        decoder_dtype: precision.decoder_dtype,
                        vision_dtype: precision.vision_dtype,
                        adapter_dtype: precision.adapter_dtype,
                    },
                    device,
                )
                .map_err(|e| candle::Error::msg(e.to_string()))?;
            let vision = VisionRuntime::Float(StarVectorVisionTransformer::new(
                loader_views.vision,
                &vision_cfg,
            )?);
            let adapter = AdapterRuntime::Float(
                StarVectorAdapter::new(loader_views.adapter, &loader)
                    .map_err(|e| candle::Error::msg(e.to_string()))?,
            );
            let decoder =
                DecoderRuntime::Float(BigCodeDecoder::load(loader_views.decoder, decoder_cfg)?);
            (vision, adapter, decoder)
        };

        let tokenizer = load_tokenizer(&model_dir)?;
        let prompt_tokens = tokenizer
            .encode("<svg", false)
            .map_err(|e| candle::Error::msg(e.to_string()))?
            .get_ids()
            .to_vec();
        let stop_tokens = tokenizer
            .encode("</svg>", false)
            .map_err(|e| candle::Error::msg(e.to_string()))?
            .get_ids()
            .to_vec();

        Ok(Self {
            device: device.clone(),
            precision,
            tokenizer,
            prompt_tokens,
            stop_tokens,
            preprocessor,
            vision,
            adapter,
            decoder,
        })
    }

    pub fn generate_svg(
        &mut self,
        image: impl AsRef<Path>,
        config: &GenerationConfig,
    ) -> candle::Result<GenerationOutput> {
        self.decoder.clear_kv_cache();

        let pixel_values = self
            .preprocessor
            .preprocess_to_tensor(image, &self.device)
            .map_err(|e| candle::Error::msg(e.to_string()))?
            .unsqueeze(0)?
            .to_dtype(self.precision.vision_dtype)?;
        let visual_tokens = self.vision.forward_sequence(&pixel_values)?;
        let visual_tokens = visual_tokens.to_dtype(self.precision.adapter_dtype)?;
        let visual_tokens = self.adapter.forward(&visual_tokens)?;
        let visual_tokens = visual_tokens.to_dtype(self.precision.decoder_dtype)?;

        let prompt_len = self.prompt_tokens.len();
        let prompt_ids = Tensor::from_slice(&self.prompt_tokens, (1, prompt_len), &self.device)?;
        let prompt_embeds = self.decoder.embed_tokens(&prompt_ids)?;
        let prefix_embeds = Tensor::cat(&[&visual_tokens, &prompt_embeds], 1)?;
        let prefix_len = prefix_embeds.dim(1)?;
        let max_ctx = self.decoder.config().max_position_embeddings;
        if prefix_len >= max_ctx {
            candle::bail!(
                "prefix length ({prefix_len}) exceeds decoder context window ({max_ctx})"
            );
        }
        let max_new_tokens_supported = max_ctx - prefix_len;
        if config.max_new_tokens > max_new_tokens_supported {
            candle::bail!(
                "max_new_tokens={} exceeds model limit for this input: {} (context_window={}, prefix_len={})",
                config.max_new_tokens,
                max_new_tokens_supported,
                max_ctx,
                prefix_len
            );
        }

        let mut logits = self.decoder.forward_inputs_embeds(&prefix_embeds, 0)?;
        let mut generated_tokens = Vec::with_capacity(config.max_new_tokens);
        let mut history_tokens = self.prompt_tokens.clone();
        let mut logits_processor = LogitsProcessor::new(
            config.seed,
            if config.do_sample {
                Some(config.temperature)
            } else {
                None
            },
            if config.do_sample {
                Some(config.top_p)
            } else {
                None
            },
        );

        for _ in 0..config.max_new_tokens {
            let next_token =
                next_token_id(&logits, &history_tokens, config, &mut logits_processor)?;
            generated_tokens.push(next_token);
            history_tokens.push(next_token);
            if has_suffix(&generated_tokens, &self.stop_tokens) {
                break;
            }
            if generated_tokens.len() == config.max_new_tokens {
                break;
            }
            let next_input = Tensor::from_slice(&[next_token], (1, 1), &self.device)?;
            let past_len = prefix_len + generated_tokens.len() - 1;
            logits = self.decoder.forward(&next_input, past_len)?;
        }
        let mut full_decode_tokens = self.prompt_tokens.clone();
        full_decode_tokens.extend_from_slice(&generated_tokens);
        let svg = self
            .tokenizer
            .decode(&full_decode_tokens, false)
            .map_err(|e| candle::Error::msg(e.to_string()))?;

        Ok(GenerationOutput {
            svg,
            token_ids: generated_tokens,
        })
    }
}

fn load_tokenizer(model_dir: &Path) -> candle::Result<Tokenizer> {
    let tokenizer_path = model_dir.join(TOKENIZER_JSON_FILE);
    let tokenizer_path = tokenizer_path
        .to_str()
        .ok_or_else(|| candle::Error::msg("tokenizer path contains invalid UTF-8"))?;
    Tokenizer::from_file(tokenizer_path).map_err(|e| candle::Error::msg(e.to_string()))
}

fn vision_config_from_metadata(metadata: &ParsedModelMetadata) -> StarVectorVisionConfig {
    let mut cfg = StarVectorVisionConfig {
        image_size: metadata.model_config.image_size,
        ..Default::default()
    };
    if let Some(scale) = metadata.model_config.hidden_size_scale
        && scale > 0
    {
        cfg.embed_dim = metadata.model_config.hidden_size / scale;
        cfg.intermediate_size = cfg.embed_dim * 4;
    }
    cfg
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use candle::{DType, Device};

    use super::{PrecisionPolicy, StarVector};
    use crate::model::generation::GenerationConfig;

    fn model_dir() -> PathBuf {
        if let Ok(path) = std::env::var("STARVECTOR_MODEL_DIR") {
            return PathBuf::from(path);
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("models")
            .join("starvector-1b-im2svg")
    }

    fn sample_image() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("star-vector")
            .join("assets")
            .join("examples")
            .join("sample-18.png")
    }

    #[test]
    fn returns_generated_text_without_repair_when_budget_is_tiny() -> candle::Result<()> {
        let device = Device::Cpu;
        let mut model = StarVector::load(model_dir(), &device, PrecisionPolicy::cpu_default())?;
        let out = model
            .generate_svg(
                sample_image(),
                &GenerationConfig {
                    max_new_tokens: 8,
                    ..Default::default()
                },
            )
            .expect("tiny budget should still return generated output");
        assert!(out.svg.starts_with("<svg"));
        Ok(())
    }

    #[test]
    fn precision_policy_for_device_matches_defaults() {
        let cpu = PrecisionPolicy::for_device(&Device::Cpu);
        assert_eq!(cpu.decoder_dtype, DType::F32);
        assert_eq!(cpu.vision_dtype, DType::F32);
        assert_eq!(cpu.adapter_dtype, DType::F32);
    }
}
