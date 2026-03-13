use candle::{D, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder, embedding, linear_b as linear};

use crate::config::ModelConfig;

fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    let weight = vb.get(size, "weight")?;
    let bias = vb.get(size, "bias")?;
    Ok(LayerNorm::new(weight, bias, eps))
}

fn make_causal_mask(t: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<_> = (0..t)
        .flat_map(|i| (0..t).map(move |j| u8::from(j <= i)))
        .collect();
    Tensor::from_slice(&mask, (t, t), device)
}

#[derive(Debug, Clone)]
pub struct BigCodeConfig {
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub layer_norm_epsilon: f64,
    pub n_inner: Option<usize>,
    pub num_attention_heads: usize,
    pub multi_query: bool,
    pub use_cache: bool,
}

impl BigCodeConfig {
    pub fn from_model_config(cfg: &ModelConfig) -> Self {
        Self {
            vocab_size: cfg.vocab_size,
            max_position_embeddings: cfg.max_position_embeddings,
            num_hidden_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            layer_norm_epsilon: 1e-5,
            n_inner: None,
            num_attention_heads: cfg.num_attention_heads,
            multi_query: cfg.multi_query,
            use_cache: cfg.use_cache.unwrap_or(true),
        }
    }
}

struct Attention {
    c_attn: Linear,
    c_proj: Linear,
    kv_cache: Option<Tensor>,
    use_cache: bool,
    embed_dim: usize,
    kv_dim: usize,
    num_heads: usize,
    head_dim: usize,
    multi_query: bool,
}

impl Attention {
    fn load(vb: VarBuilder, cfg: &BigCodeConfig) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let head_dim = hidden_size / cfg.num_attention_heads;
        let kv_heads = if cfg.multi_query {
            1
        } else {
            cfg.num_attention_heads
        };
        let kv_dim = kv_heads * head_dim;
        let c_attn = linear(hidden_size, hidden_size + 2 * kv_dim, true, vb.pp("c_attn"))?;
        let c_proj = linear(hidden_size, hidden_size, true, vb.pp("c_proj"))?;
        Ok(Self {
            c_proj,
            c_attn,
            kv_cache: None,
            use_cache: cfg.use_cache,
            embed_dim: hidden_size,
            kv_dim,
            num_heads: cfg.num_attention_heads,
            head_dim,
            multi_query: cfg.multi_query,
        })
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }

    fn attn(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        if query.dtype() != DType::F32 {
            candle::bail!("upcasting is not supported {:?}", query.dtype())
        }
        let scale_factor = 1f64 / (self.head_dim as f64).sqrt();
        let initial_query_shape = query.shape();
        let key_len = key.dim(D::Minus1)?;
        let (query, key, attn_shape, attn_view) = if self.multi_query {
            let (b_sz, query_len, _) = query.dims3()?;
            let query = query.reshape((b_sz, query_len * self.num_heads, self.head_dim))?;
            let attn_shape = (b_sz, query_len, self.num_heads, key_len);
            let attn_view = (b_sz, query_len * self.num_heads, key_len);
            (query, key.clone(), attn_shape, attn_view)
        } else {
            let (b_sz, _num_heads, query_len, _head_dim) = query.dims4()?;
            let query = query.reshape((b_sz, query_len * self.num_heads, self.head_dim))?;
            let key = key.reshape((b_sz * self.num_heads, self.head_dim, key_len))?;
            let attn_shape = (b_sz, self.num_heads, query_len, key_len);
            let attn_view = (b_sz * self.num_heads, query_len, key_len);
            (query, key, attn_shape, attn_view)
        };

        let attn_weights =
            (query.matmul(&key.contiguous()?)? * scale_factor)?.reshape(attn_shape)?;
        let attention_mask = attention_mask.broadcast_as(attn_shape)?;
        let mask_value = Tensor::new(f32::NEG_INFINITY, query.device())?
            .to_dtype(attn_weights.dtype())?
            .broadcast_as(attn_shape)?;
        let attn_weights = attention_mask.where_cond(&attn_weights, &mask_value)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let value = value.contiguous()?;
        let attn_output = if self.multi_query {
            attn_weights
                .reshape(attn_view)?
                .matmul(&value)?
                .reshape(initial_query_shape)?
        } else {
            attn_weights.matmul(&value)?
        };
        Ok(attn_output)
    }

    fn forward(&mut self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let qkv = self.c_attn.forward(hidden_states)?;
        let (query, mut key_value) = if self.multi_query {
            let query = qkv.i((.., .., ..self.embed_dim))?;
            let key_value = qkv.i((.., .., self.embed_dim..self.embed_dim + 2 * self.kv_dim))?;
            (query, key_value)
        } else {
            let mut dims = qkv.dims().to_vec();
            dims.pop();
            dims.push(self.embed_dim);
            dims.push(self.head_dim * 3);
            let qkv = qkv.reshape(dims)?.transpose(1, 2)?;
            let query = qkv.i((.., .., .., ..self.head_dim))?;
            let key_value = qkv.i((.., .., .., self.head_dim..3 * self.head_dim))?;
            (query, key_value)
        };

        if self.use_cache {
            if let Some(kv_cache) = &self.kv_cache {
                key_value = Tensor::cat(&[kv_cache, &key_value], D::Minus2)?.contiguous()?;
            }
            self.kv_cache = Some(key_value.clone());
        }

        let key = key_value.narrow(D::Minus1, 0, self.head_dim)?;
        let value = key_value.narrow(D::Minus1, self.head_dim, self.head_dim)?;
        let attn_output = self.attn(&query, &key.t()?, &value, attention_mask)?;
        let attn_output = if self.multi_query {
            attn_output
        } else {
            attn_output
                .transpose(1, 2)?
                .reshape(hidden_states.shape())?
        };
        self.c_proj.forward(&attn_output)
    }
}

struct Mlp {
    c_fc: Linear,
    c_proj: Linear,
}

impl Mlp {
    fn load(inner_dim: usize, vb: VarBuilder, cfg: &BigCodeConfig) -> Result<Self> {
        let c_fc = linear(cfg.hidden_size, inner_dim, true, vb.pp("c_fc"))?;
        let c_proj = linear(inner_dim, cfg.hidden_size, true, vb.pp("c_proj"))?;
        Ok(Self { c_fc, c_proj })
    }

    fn forward(&mut self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.c_fc.forward(hidden_states)?.gelu()?;
        self.c_proj.forward(&hidden_states)
    }
}

struct Block {
    ln_1: LayerNorm,
    attn: Attention,
    ln_2: LayerNorm,
    mlp: Mlp,
}

impl Block {
    fn load(vb: VarBuilder, cfg: &BigCodeConfig) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let inner_dim = cfg.n_inner.unwrap_or(4 * hidden_size);
        let ln_1 = layer_norm(hidden_size, cfg.layer_norm_epsilon, vb.pp("ln_1"))?;
        let attn = Attention::load(vb.pp("attn"), cfg)?;
        let ln_2 = layer_norm(hidden_size, cfg.layer_norm_epsilon, vb.pp("ln_2"))?;
        let mlp = Mlp::load(inner_dim, vb.pp("mlp"), cfg)?;
        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        })
    }

    fn clear_kv_cache(&mut self) {
        self.attn.clear_kv_cache();
    }

    fn forward(&mut self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let residual = hidden_states;
        let hidden_states = self.ln_1.forward(hidden_states)?;
        let attn_outputs = self.attn.forward(&hidden_states, attention_mask)?;
        let hidden_states = (&attn_outputs + residual)?;
        let residual = &hidden_states;
        let hidden_states = self.ln_2.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        &hidden_states + residual
    }
}

pub struct BigCodeDecoder {
    wte: Embedding,
    wpe: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    lm_head: Linear,
    bias: Tensor,
    config: BigCodeConfig,
}

impl BigCodeDecoder {
    pub fn load(vb: VarBuilder, cfg: BigCodeConfig) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let vb_t = vb.pp("transformer");
        let wte = embedding(cfg.vocab_size, hidden_size, vb_t.pp("wte"))?;
        let wpe = embedding(cfg.max_position_embeddings, hidden_size, vb_t.pp("wpe"))?;
        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(vb_t.pp(format!("h.{i}")), &cfg))
            .collect::<Result<Vec<_>>>()?;
        let ln_f = layer_norm(hidden_size, cfg.layer_norm_epsilon, vb_t.pp("ln_f"))?;
        // tied weights: lm_head uses transformer.wte
        let lm_head = linear(hidden_size, cfg.vocab_size, false, vb_t.pp("wte"))?;
        let bias = make_causal_mask(cfg.max_position_embeddings, vb.device())?;
        Ok(Self {
            wte,
            wpe,
            blocks,
            ln_f,
            lm_head,
            bias,
            config: cfg,
        })
    }

    pub fn config(&self) -> &BigCodeConfig {
        &self.config
    }

    pub fn token_embedding(&self) -> &Embedding {
        &self.wte
    }

    pub fn clear_kv_cache(&mut self) {
        for block in &mut self.blocks {
            block.clear_kv_cache();
        }
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.wte.forward(input_ids)
    }

    pub fn forward_inputs_embeds(
        &mut self,
        input_embeds: &Tensor,
        past_len: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = input_embeds.dims3()?;
        let dev = input_embeds.device();
        let key_len = past_len + seq_len;
        let attention_mask = self.bias.i((past_len..key_len, ..key_len))?.unsqueeze(0)?;
        let seq_len_dim = if self.config.multi_query { 2 } else { 1 };
        let attention_mask = attention_mask.unsqueeze(seq_len_dim)?;

        let position_ids = Tensor::arange(past_len as u32, (past_len + seq_len) as u32, dev)?;
        let position_ids = position_ids.unsqueeze(0)?.broadcast_as((b_sz, seq_len))?;
        let position_embeds = self.wpe.forward(&position_ids)?;

        let mut hidden_states = (input_embeds + &position_embeds)?;
        for block in &mut self.blocks {
            hidden_states = block.forward(&hidden_states, &attention_mask)?;
        }
        let hidden_states = self.ln_f.forward(&hidden_states)?;
        let hidden_states = hidden_states
            .reshape((b_sz, seq_len, self.config.hidden_size))?
            .narrow(1, seq_len - 1, 1)?;
        self.lm_head.forward(&hidden_states)?.squeeze(1)
    }

    pub fn forward(&mut self, input_ids: &Tensor, past_len: usize) -> Result<Tensor> {
        let input_embeds = self.embed_tokens(input_ids)?;
        self.forward_inputs_embeds(&input_embeds, past_len)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use candle::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    use super::{BigCodeConfig, BigCodeDecoder};
    use crate::model::loader::{LoaderPrecisionPolicy, ReuseFirstWeightLoader};
    use crate::types::ParsedModelMetadata;

    fn tiny_config() -> BigCodeConfig {
        BigCodeConfig {
            vocab_size: 32,
            max_position_embeddings: 32,
            num_hidden_layers: 2,
            hidden_size: 16,
            layer_norm_epsilon: 1e-5,
            n_inner: Some(32),
            num_attention_heads: 4,
            multi_query: true,
            use_cache: true,
        }
    }

    fn model_dir() -> PathBuf {
        if let Ok(path) = std::env::var("STARVECTOR_MODEL_DIR") {
            return PathBuf::from(path);
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("models")
            .join("starvector-1b-im2svg")
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> candle::Result<f32> {
        let diff = (a - b)?.abs()?;
        let vals = diff.flatten_all()?.to_vec1::<f32>()?;
        Ok(vals.into_iter().fold(0.0_f32, f32::max))
    }

    #[test]
    fn embed_tokens_and_inputs_embeds_paths_are_consistent() -> candle::Result<()> {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let mut decoder = BigCodeDecoder::load(vb, tiny_config())?;

        let ids = Tensor::from_slice(&[1_u32, 2, 3, 4], (1, 4), &dev)?;
        let embeds = decoder.embed_tokens(&ids)?;
        assert_eq!(embeds.dims3()?, (1, 4, tiny_config().hidden_size));

        decoder.clear_kv_cache();
        let logits_from_ids = decoder.forward(&ids, 0)?;
        decoder.clear_kv_cache();
        let logits_from_embeds = decoder.forward_inputs_embeds(&embeds, 0)?;
        assert_eq!(logits_from_ids.dims2()?, (1, tiny_config().vocab_size));
        assert_eq!(logits_from_embeds.dims2()?, (1, tiny_config().vocab_size));
        assert_eq!(max_abs_diff(&logits_from_ids, &logits_from_embeds)?, 0.0);
        Ok(())
    }

    #[test]
    fn prefix_and_cached_step_match_full_recompute_on_real_weights() -> candle::Result<()> {
        let parsed = ParsedModelMetadata::from_model_dir(&model_dir())
            .map_err(|e| candle::Error::msg(e.to_string()))?;
        let cfg = BigCodeConfig::from_model_config(&parsed.model_config);
        let loader = ReuseFirstWeightLoader::from_model_dir(model_dir())
            .map_err(|e| candle::Error::msg(e.to_string()))?;
        let views = loader
            .make_views(LoaderPrecisionPolicy::cpu_default(), &Device::Cpu)
            .map_err(|e| candle::Error::msg(e.to_string()))?;
        let mut decoder = BigCodeDecoder::load(views.decoder, cfg)?;

        let prefix_ids = Tensor::from_slice(&[46_u32, 3672, 83], (1, 3), &Device::Cpu)?;
        let next_id = Tensor::from_slice(&[34_u32], (1, 1), &Device::Cpu)?;
        let full_ids = Tensor::from_slice(&[46_u32, 3672, 83, 34], (1, 4), &Device::Cpu)?;

        decoder.clear_kv_cache();
        let prefix_embeds = decoder.embed_tokens(&prefix_ids)?;
        let _prefix_logits = decoder.forward_inputs_embeds(&prefix_embeds, 0)?;
        let cached_step_logits = decoder.forward(&next_id, 3)?;

        decoder.clear_kv_cache();
        let full_recompute_logits = decoder.forward(&full_ids, 0)?;

        assert!(max_abs_diff(&cached_step_logits, &full_recompute_logits)? < 5e-3);
        Ok(())
    }

    #[test]
    fn clear_kv_cache_restores_fresh_prefix_behavior() -> candle::Result<()> {
        let parsed = ParsedModelMetadata::from_model_dir(&model_dir())
            .map_err(|e| candle::Error::msg(e.to_string()))?;
        let cfg = BigCodeConfig::from_model_config(&parsed.model_config);
        let loader = ReuseFirstWeightLoader::from_model_dir(model_dir())
            .map_err(|e| candle::Error::msg(e.to_string()))?;
        let views = loader
            .make_views(LoaderPrecisionPolicy::cpu_default(), &Device::Cpu)
            .map_err(|e| candle::Error::msg(e.to_string()))?;
        let mut decoder = BigCodeDecoder::load(views.decoder, cfg)?;

        let ids = Tensor::from_slice(&[46_u32, 3672], (1, 2), &Device::Cpu)?;

        decoder.clear_kv_cache();
        let first = decoder.forward(&ids, 0)?;
        let _dirty = decoder.forward(&ids, 0);
        decoder.clear_kv_cache();
        let second = decoder.forward(&ids, 0)?;

        assert!(max_abs_diff(&first, &second)? < 1e-6);
        Ok(())
    }
}
