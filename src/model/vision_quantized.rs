use std::collections::HashMap;
use std::sync::Arc;

use candle::quantized::{QMatMul, QTensor, gguf_file};
use candle::{DType, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, Module};

use crate::model::vision::StarVectorVisionConfig;

#[derive(Clone)]
struct QuantStore {
    tensors: Arc<HashMap<String, Arc<QTensor>>>,
    device: candle::Device,
}

impl QuantStore {
    fn from_gguf(path: &std::path::Path, device: &candle::Device) -> Result<Self> {
        const ROOT: &str = "model.image_encoder.";
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut tensors = HashMap::new();
        for name in content.tensor_infos.keys() {
            if !name.starts_with(ROOT) {
                continue;
            }
            if content
                .tensor_infos
                .get(name)
                .is_some_and(|info| info.shape.rank() == 0)
            {
                continue;
            }
            let t = content.tensor(&mut file, name, device)?;
            tensors.insert(name.clone(), Arc::new(t));
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

#[derive(Clone, Debug)]
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
        let in_dtype = x.dtype();
        let x = if in_dtype != DType::F32 {
            x.to_dtype(DType::F32)?
        } else {
            x.clone()
        };
        let x = x.apply(&self.weight)?;
        let x = match &self.bias {
            None => x,
            Some(bias) => x.broadcast_add(bias)?,
        };
        if in_dtype != DType::F32 {
            x.to_dtype(in_dtype)
        } else {
            Ok(x)
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
    let (weight, bias) = if store.device.is_cuda() {
        (weight.to_dtype(DType::F16)?, bias.to_dtype(DType::F16)?)
    } else {
        (weight, bias)
    };
    Ok(LayerNorm::new(weight, bias, eps))
}

#[derive(Clone, Debug)]
struct VisionEmbeddings {
    patch_embedding: Conv2d,
    class_embedding: Tensor,
    position_embedding: Tensor,
}

impl VisionEmbeddings {
    fn new(store: &QuantStore, cfg: &StarVectorVisionConfig) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let patch_w = store
            .get_with_shape(
                "model.image_encoder.visual_encoder.conv1.weight",
                &[
                    cfg.embed_dim,
                    cfg.num_channels,
                    cfg.patch_size,
                    cfg.patch_size,
                ],
            )?
            .dequantize(&store.device)?;
        let patch_w = if store.device.is_cuda() {
            patch_w.to_dtype(DType::F16)?
        } else {
            patch_w
        };
        let patch_embedding = Conv2d::new(patch_w, None, conv_cfg);
        let class_embedding = store
            .get_with_shape(
                "model.image_encoder.visual_encoder.class_embedding",
                &[cfg.embed_dim],
            )?
            .dequantize(&store.device)?;
        let position_embedding = store
            .get_with_shape(
                "model.image_encoder.visual_encoder.positional_embedding",
                &[cfg.num_positions(), cfg.embed_dim],
            )?
            .dequantize(&store.device)?;
        Ok(Self {
            patch_embedding,
            class_embedding,
            position_embedding,
        })
    }
}

impl Module for VisionEmbeddings {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let batch_size = xs.dim(0)?;
        let patch = self
            .patch_embedding
            .forward(xs)?
            .flatten_from(2)?
            .transpose(1, 2)?;

        let embed_dim = self.class_embedding.dim(0)?;
        let class = self
            .class_embedding
            .to_dtype(dtype)?
            .reshape((1, 1, embed_dim))?
            .expand((batch_size, 1, embed_dim))?;
        let embeddings = Tensor::cat(&[class, patch], 1)?;
        let pos = self.position_embedding.to_dtype(dtype)?;
        embeddings.broadcast_add(&pos)
    }
}

#[derive(Clone, Debug)]
struct Attention {
    in_proj: QuantLinear,
    out_proj: QuantLinear,
    num_attention_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    fn new(store: &QuantStore, base: &str, cfg: &StarVectorVisionConfig) -> Result<Self> {
        let embed_dim = cfg.embed_dim;
        let in_proj_w = store.get_with_shape(
            &format!("{base}.attn.in_proj_weight"),
            &[embed_dim * 3, embed_dim],
        )?;
        let in_proj_b = store
            .get_with_shape(&format!("{base}.attn.in_proj_bias"), &[embed_dim * 3])?
            .dequantize(&store.device)?;
        let in_proj = QuantLinear::from_qtensor(in_proj_w, Some(in_proj_b))?;

        let out_w = store.get_with_shape(
            &format!("{base}.attn.out_proj.weight"),
            &[embed_dim, embed_dim],
        )?;
        let out_b = store
            .get_with_shape(&format!("{base}.attn.out_proj.bias"), &[embed_dim])?
            .dequantize(&store.device)?;
        let out_proj = QuantLinear::from_qtensor(out_w, Some(out_b))?;

        let head_dim = embed_dim / cfg.num_attention_heads;
        Ok(Self {
            in_proj,
            out_proj,
            num_attention_heads: cfg.num_attention_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    fn shape_multi_head(&self, xs: &Tensor, bsz: usize, seq_len: usize) -> Result<Tensor> {
        xs.reshape((bsz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(DType::F32)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let in_dtype = xs.dtype();
        let (bsz, seq_len, embed_dim) = xs.dims3()?;
        let qkv = self.in_proj.forward(xs)?;
        let q = self.shape_multi_head(&qkv.narrow(2, 0, embed_dim)?, bsz, seq_len)?;
        let k = self.shape_multi_head(&qkv.narrow(2, embed_dim, embed_dim)?, bsz, seq_len)?;
        let v = self.shape_multi_head(&qkv.narrow(2, embed_dim * 2, embed_dim)?, bsz, seq_len)?;
        let q = (q * self.scale)?;
        let attn = q.matmul(&k.transpose(2, 3)?)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?.to_dtype(in_dtype)?;
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((bsz, seq_len, embed_dim))?;
        self.out_proj.forward(&out)
    }
}

#[derive(Clone, Debug)]
struct Mlp {
    fc1: QuantLinear,
    fc2: QuantLinear,
}

impl Mlp {
    fn new(store: &QuantStore, base: &str, cfg: &StarVectorVisionConfig) -> Result<Self> {
        let fc1_w = store.get_with_shape(
            &format!("{base}.mlp.c_fc.weight"),
            &[cfg.intermediate_size, cfg.embed_dim],
        )?;
        let fc1_b = store
            .get_with_shape(&format!("{base}.mlp.c_fc.bias"), &[cfg.intermediate_size])?
            .dequantize(&store.device)?;
        let fc1 = QuantLinear::from_qtensor(fc1_w, Some(fc1_b))?;

        let fc2_w = store.get_with_shape(
            &format!("{base}.mlp.c_proj.weight"),
            &[cfg.embed_dim, cfg.intermediate_size],
        )?;
        let fc2_b = store
            .get_with_shape(&format!("{base}.mlp.c_proj.bias"), &[cfg.embed_dim])?
            .dequantize(&store.device)?;
        let fc2 = QuantLinear::from_qtensor(fc2_w, Some(fc2_b))?;
        Ok(Self { fc1, fc2 })
    }

    fn quick_gelu(xs: &Tensor) -> Result<Tensor> {
        xs * candle_nn::ops::sigmoid(&(xs * 1.702f64)?)?
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        let xs = Self::quick_gelu(&xs)?;
        self.fc2.forward(&xs)
    }
}

#[derive(Clone, Debug)]
struct EncoderLayer {
    attn: Attention,
    ln_1: LayerNorm,
    mlp: Mlp,
    ln_2: LayerNorm,
}

impl EncoderLayer {
    fn new(store: &QuantStore, idx: usize, cfg: &StarVectorVisionConfig) -> Result<Self> {
        let base = format!("model.image_encoder.visual_encoder.transformer.resblocks.{idx}");
        let attn = Attention::new(store, &base, cfg)?;
        let ln_1 = layer_norm(store, &format!("{base}.ln_1"), cfg.embed_dim, 1e-5)?;
        let mlp = Mlp::new(store, &base, cfg)?;
        let ln_2 = layer_norm(store, &format!("{base}.ln_2"), cfg.embed_dim, 1e-5)?;
        Ok(Self {
            attn,
            ln_1,
            mlp,
            ln_2,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.ln_1.forward(xs)?;
        let xs = self.attn.forward(&xs)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.ln_2.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        xs + residual
    }
}

#[derive(Clone, Debug)]
pub struct StarVectorVisionTransformerQuantized {
    embeddings: VisionEmbeddings,
    pre_layer_norm: LayerNorm,
    layers: Vec<EncoderLayer>,
    post_layer_norm: LayerNorm,
}

impl StarVectorVisionTransformerQuantized {
    pub fn load_from_gguf(
        path: &std::path::Path,
        device: &candle::Device,
        cfg: &StarVectorVisionConfig,
    ) -> Result<Self> {
        let store = QuantStore::from_gguf(path, device)?;
        let embeddings = VisionEmbeddings::new(&store, cfg)?;
        let pre_layer_norm = layer_norm(
            &store,
            "model.image_encoder.visual_encoder.ln_pre",
            cfg.embed_dim,
            1e-5,
        )?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(EncoderLayer::new(&store, i, cfg)?);
        }
        let post_layer_norm =
            layer_norm(&store, "model.image_encoder.ln_vision", cfg.embed_dim, 1e-5)?;
        Ok(Self {
            embeddings,
            pre_layer_norm,
            layers,
            post_layer_norm,
        })
    }

    pub fn forward_sequence(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let mut xs = self.embeddings.forward(pixel_values)?;
        xs = self.pre_layer_norm.forward(&xs)?;
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        self.post_layer_norm.forward(&xs)
    }
}

impl Module for StarVectorVisionTransformerQuantized {
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        self.forward_sequence(pixel_values)
    }
}
