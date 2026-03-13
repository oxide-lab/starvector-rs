use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use candle::quantized::{QMatMul, QTensor, gguf_file};
use candle::{D, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Module};

use crate::model::bigcode::BigCodeConfig;

fn make_causal_mask(t: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<_> = (0..t)
        .flat_map(|i| (0..t).map(move |j| u8::from(j <= i)))
        .collect();
    Tensor::from_slice(&mask, (t, t), device)
}

#[derive(Clone)]
struct QuantStore {
    tensors: Arc<HashMap<String, Arc<QTensor>>>,
    device: Device,
}

impl QuantStore {
    fn from_gguf(path: &Path, device: &Device) -> Result<Self> {
        const DECODER_ROOT: &str = "model.svg_transformer.transformer.transformer.";
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut tensors = HashMap::with_capacity(content.tensor_infos.len());
        for name in content.tensor_infos.keys() {
            if !name.starts_with(DECODER_ROOT) {
                continue;
            }
            if content
                .tensor_infos
                .get(name)
                .is_some_and(|info| info.shape.rank() == 0)
            {
                continue;
            }
            let tensor = content.tensor(&mut file, name, device)?;
            tensors.insert(name.to_string(), Arc::new(tensor));
        }
        Ok(Self {
            tensors: Arc::new(tensors),
            device: device.clone(),
        })
    }

    fn get(&self, name: &str) -> Result<Arc<QTensor>> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| candle::Error::msg(format!("cannot find tensor {name}")))
    }

    fn get_with_shape(&self, name: &str, expected: &[usize]) -> Result<Arc<QTensor>> {
        let t = self.get(name)?;
        if t.shape().dims() != expected {
            candle::bail!(
                "shape mismatch for {name}, got {:?}, expected {:?}",
                t.shape(),
                expected
            )
        }
        Ok(t)
    }
}

#[derive(Debug, Clone)]
struct QuantLinear {
    weight: QMatMul,
    bias: Option<Tensor>,
}

impl QuantLinear {
    fn from_qtensor(weight: Arc<QTensor>, bias: Option<Tensor>) -> Result<Self> {
        let weight = QMatMul::from_arc(weight)?;
        Ok(Self { weight, bias })
    }
}

impl Module for QuantLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.apply(&self.weight)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

fn layer_norm(store: &QuantStore, base: &str, size: usize, eps: f64) -> Result<LayerNorm> {
    let weight = store
        .get_with_shape(&format!("{base}.weight"), &[size])?
        .dequantize(&store.device)?;
    let bias = store
        .get_with_shape(&format!("{base}.bias"), &[size])?
        .dequantize(&store.device)?;
    Ok(LayerNorm::new(weight, bias, eps))
}

struct Attention {
    c_attn: QuantLinear,
    c_proj: QuantLinear,
    kv_cache: Option<Tensor>,
    use_cache: bool,
    embed_dim: usize,
    kv_dim: usize,
    num_heads: usize,
    head_dim: usize,
    multi_query: bool,
}

impl Attention {
    fn load(store: &QuantStore, base: &str, cfg: &BigCodeConfig) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let head_dim = hidden_size / cfg.num_attention_heads;
        let kv_heads = if cfg.multi_query {
            1
        } else {
            cfg.num_attention_heads
        };
        let kv_dim = kv_heads * head_dim;

        let c_attn_weight = store.get_with_shape(
            &format!("{base}.c_attn.weight"),
            &[hidden_size + 2 * kv_dim, hidden_size],
        )?;
        let c_attn_bias = store
            .get_with_shape(&format!("{base}.c_attn.bias"), &[hidden_size + 2 * kv_dim])?
            .dequantize(&store.device)?;
        let c_attn = QuantLinear::from_qtensor(c_attn_weight, Some(c_attn_bias))?;

        let c_proj_weight = store.get_with_shape(
            &format!("{base}.c_proj.weight"),
            &[hidden_size, hidden_size],
        )?;
        let c_proj_bias = store
            .get_with_shape(&format!("{base}.c_proj.bias"), &[hidden_size])?
            .dequantize(&store.device)?;
        let c_proj = QuantLinear::from_qtensor(c_proj_weight, Some(c_proj_bias))?;

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
    c_fc: QuantLinear,
    c_proj: QuantLinear,
}

impl Mlp {
    fn load(inner_dim: usize, store: &QuantStore, base: &str, cfg: &BigCodeConfig) -> Result<Self> {
        let c_fc_weight = store.get_with_shape(
            &format!("{base}.c_fc.weight"),
            &[inner_dim, cfg.hidden_size],
        )?;
        let c_fc_bias = store
            .get_with_shape(&format!("{base}.c_fc.bias"), &[inner_dim])?
            .dequantize(&store.device)?;
        let c_fc = QuantLinear::from_qtensor(c_fc_weight, Some(c_fc_bias))?;

        let c_proj_weight = store.get_with_shape(
            &format!("{base}.c_proj.weight"),
            &[cfg.hidden_size, inner_dim],
        )?;
        let c_proj_bias = store
            .get_with_shape(&format!("{base}.c_proj.bias"), &[cfg.hidden_size])?
            .dequantize(&store.device)?;
        let c_proj = QuantLinear::from_qtensor(c_proj_weight, Some(c_proj_bias))?;
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
    fn load(store: &QuantStore, base: &str, cfg: &BigCodeConfig) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let inner_dim = cfg.n_inner.unwrap_or(4 * hidden_size);
        let ln_1 = layer_norm(
            store,
            &format!("{base}.ln_1"),
            hidden_size,
            cfg.layer_norm_epsilon,
        )?;
        let attn = Attention::load(store, &format!("{base}.attn"), cfg)?;
        let ln_2 = layer_norm(
            store,
            &format!("{base}.ln_2"),
            hidden_size,
            cfg.layer_norm_epsilon,
        )?;
        let mlp = Mlp::load(inner_dim, store, &format!("{base}.mlp"), cfg)?;
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

pub struct BigCodeDecoderQuantized {
    wte: Embedding,
    wpe: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    lm_head: QuantLinear,
    bias: Tensor,
    config: BigCodeConfig,
}

impl BigCodeDecoderQuantized {
    pub fn load_from_gguf(path: &Path, device: &Device, cfg: BigCodeConfig) -> Result<Self> {
        let store = QuantStore::from_gguf(path, device)?;
        Self::load(store, cfg)
    }

    fn load(store: QuantStore, cfg: BigCodeConfig) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let root = "model.svg_transformer.transformer.transformer";
        let wte_weight = store
            .get_with_shape(
                &format!("{root}.wte.weight"),
                &[cfg.vocab_size, hidden_size],
            )?
            .dequantize(&store.device)?;
        let wte = Embedding::new(wte_weight, hidden_size);
        let wpe_weight = store
            .get_with_shape(
                &format!("{root}.wpe.weight"),
                &[cfg.max_position_embeddings, hidden_size],
            )?
            .dequantize(&store.device)?;
        let wpe = Embedding::new(wpe_weight, hidden_size);
        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(&store, &format!("{root}.h.{i}"), &cfg))
            .collect::<Result<Vec<_>>>()?;
        let ln_f = layer_norm(
            &store,
            &format!("{root}.ln_f"),
            hidden_size,
            cfg.layer_norm_epsilon,
        )?;
        let lm_head = QuantLinear::from_qtensor(
            store.get_with_shape(
                &format!("{root}.wte.weight"),
                &[cfg.vocab_size, hidden_size],
            )?,
            None,
        )?;
        let bias = make_causal_mask(cfg.max_position_embeddings, &store.device)?;
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
