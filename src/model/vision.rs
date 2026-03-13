use candle::{DType, Result, Tensor};
use candle_nn::{
    Conv2dConfig, LayerNorm, Linear, Module, VarBuilder, conv2d_no_bias, layer_norm, linear,
    ops::softmax_last_dim,
};

#[derive(Debug, Clone)]
pub struct StarVectorVisionConfig {
    pub embed_dim: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
}

impl Default for StarVectorVisionConfig {
    fn default() -> Self {
        Self {
            embed_dim: 1024,
            intermediate_size: 4096,
            num_hidden_layers: 23,
            num_attention_heads: 16,
            num_channels: 3,
            image_size: 224,
            patch_size: 14,
        }
    }
}

impl StarVectorVisionConfig {
    pub fn num_positions(&self) -> usize {
        let num_patches = (self.image_size / self.patch_size).pow(2);
        num_patches + 1
    }
}

pub fn map_candle_vision_name_to_checkpoint(name: &str) -> String {
    let name = name.strip_prefix("model.image_encoder.").unwrap_or(name);

    if name == "embeddings.class_embedding" {
        return "visual_encoder.class_embedding".to_owned();
    }
    if name == "embeddings.position_embedding.weight" {
        return "visual_encoder.positional_embedding".to_owned();
    }
    if let Some(rest) = name.strip_prefix("embeddings.patch_embedding.") {
        return format!("visual_encoder.conv1.{rest}");
    }
    if let Some(rest) = name.strip_prefix("pre_layrnorm.") {
        return format!("visual_encoder.ln_pre.{rest}");
    }
    if let Some(rest) = name.strip_prefix("encoder.layers.") {
        return format!("visual_encoder.transformer.resblocks.{rest}");
    }
    if let Some(rest) = name.strip_prefix("post_layernorm.") {
        return format!("ln_vision.{rest}");
    }
    name.to_owned()
}

#[derive(Clone, Debug)]
struct VisionEmbeddings {
    patch_embedding: candle_nn::Conv2d,
    class_embedding: Tensor,
    position_embedding: Tensor,
}

impl VisionEmbeddings {
    fn new(vs: VarBuilder, cfg: &StarVectorVisionConfig) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let patch_embedding = conv2d_no_bias(
            cfg.num_channels,
            cfg.embed_dim,
            cfg.patch_size,
            conv_cfg,
            vs.pp("embeddings.patch_embedding"),
        )?;
        let class_embedding = vs.get(cfg.embed_dim, "embeddings.class_embedding")?;
        let position_embedding = vs.get(
            (cfg.num_positions(), cfg.embed_dim),
            "embeddings.position_embedding.weight",
        )?;
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
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_attention_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    fn new(vs: VarBuilder, cfg: &StarVectorVisionConfig) -> Result<Self> {
        let embed_dim = cfg.embed_dim;
        let num_attention_heads = cfg.num_attention_heads;
        let in_proj = vs
            .get((embed_dim * 3, embed_dim), "attn.in_proj_weight")?
            .chunk(3, 0)?;
        let in_bias = vs.get(embed_dim * 3, "attn.in_proj_bias")?.chunk(3, 0)?;

        let q_proj = Linear::new(in_proj[0].clone(), Some(in_bias[0].clone()));
        let k_proj = Linear::new(in_proj[1].clone(), Some(in_bias[1].clone()));
        let v_proj = Linear::new(in_proj[2].clone(), Some(in_bias[2].clone()));
        let out_proj = linear(embed_dim, embed_dim, vs.pp("attn.out_proj"))?;
        let head_dim = embed_dim / num_attention_heads;
        let scale = (head_dim as f64).powf(-0.5);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_attention_heads,
            head_dim,
            scale,
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

        let q = self.shape_multi_head(&self.q_proj.forward(xs)?, bsz, seq_len)?;
        let k = self.shape_multi_head(&self.k_proj.forward(xs)?, bsz, seq_len)?;
        let v = self.shape_multi_head(&self.v_proj.forward(xs)?, bsz, seq_len)?;
        let q = (q * self.scale)?;

        let attn = q.matmul(&k.transpose(2, 3)?)?;
        let attn = softmax_last_dim(&attn)?;
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
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn new(vs: VarBuilder, cfg: &StarVectorVisionConfig) -> Result<Self> {
        let fc1 = linear(cfg.embed_dim, cfg.intermediate_size, vs.pp("mlp.c_fc"))?;
        let fc2 = linear(cfg.intermediate_size, cfg.embed_dim, vs.pp("mlp.c_proj"))?;
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
    fn new(vs: VarBuilder, cfg: &StarVectorVisionConfig) -> Result<Self> {
        let attn = Attention::new(vs.clone(), cfg)?;
        let ln_1 = layer_norm(cfg.embed_dim, 1e-5, vs.pp("ln_1"))?;
        let mlp = Mlp::new(vs.clone(), cfg)?;
        let ln_2 = layer_norm(cfg.embed_dim, 1e-5, vs.pp("ln_2"))?;
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
struct Encoder {
    layers: Vec<EncoderLayer>,
}

impl Encoder {
    fn new(vs: VarBuilder, cfg: &StarVectorVisionConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vs = vs.pp("encoder.layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            layers.push(EncoderLayer::new(vs.pp(layer_idx.to_string()), cfg)?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        Ok(xs)
    }
}

#[derive(Clone, Debug)]
pub struct StarVectorVisionTransformer {
    embeddings: VisionEmbeddings,
    pre_layer_norm: LayerNorm,
    encoder: Encoder,
    post_layer_norm: LayerNorm,
}

impl StarVectorVisionTransformer {
    pub fn new(vs: VarBuilder, cfg: &StarVectorVisionConfig) -> Result<Self> {
        // Explicit naming reconciliation between Candle-style logical path and checkpoint path.
        let vs = vs.rename_f(map_candle_vision_name_to_checkpoint);
        let embeddings = VisionEmbeddings::new(vs.clone(), cfg)?;
        let pre_layer_norm = layer_norm(cfg.embed_dim, 1e-5, vs.pp("pre_layrnorm"))?;
        let encoder = Encoder::new(vs.clone(), cfg)?;
        let post_layer_norm = layer_norm(cfg.embed_dim, 1e-5, vs.pp("post_layernorm"))?;

        Ok(Self {
            embeddings,
            pre_layer_norm,
            encoder,
            post_layer_norm,
        })
    }

    pub fn forward_sequence(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let xs = self.embeddings.forward(pixel_values)?;
        let xs = self.pre_layer_norm.forward(&xs)?;
        let xs = self.encoder.forward(&xs)?;
        self.post_layer_norm.forward(&xs)
    }
}

impl Module for StarVectorVisionTransformer {
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        self.forward_sequence(pixel_values)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use candle::{DType, Device, Tensor};

    use super::{
        StarVectorVisionConfig, StarVectorVisionTransformer, map_candle_vision_name_to_checkpoint,
    };
    use crate::model::loader::{LoaderPrecisionPolicy, ReuseFirstWeightLoader};

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
    fn name_mapping_matches_checkpoint_layout() {
        assert_eq!(
            map_candle_vision_name_to_checkpoint("embeddings.patch_embedding.weight"),
            "visual_encoder.conv1.weight"
        );
        assert_eq!(
            map_candle_vision_name_to_checkpoint("embeddings.class_embedding"),
            "visual_encoder.class_embedding"
        );
        assert_eq!(
            map_candle_vision_name_to_checkpoint("pre_layrnorm.weight"),
            "visual_encoder.ln_pre.weight"
        );
        assert_eq!(
            map_candle_vision_name_to_checkpoint("encoder.layers.7.attn.in_proj_weight"),
            "visual_encoder.transformer.resblocks.7.attn.in_proj_weight"
        );
        assert_eq!(
            map_candle_vision_name_to_checkpoint("post_layernorm.bias"),
            "ln_vision.bias"
        );
    }

    #[test]
    fn vision_forward_returns_full_sequence() -> candle::Result<()> {
        let loader = ReuseFirstWeightLoader::from_model_dir(model_dir())
            .map_err(|e| candle::Error::msg(e.to_string()))?;
        let views = loader
            .make_views(LoaderPrecisionPolicy::cpu_default(), &Device::Cpu)
            .map_err(|e| candle::Error::msg(e.to_string()))?;
        let vision =
            StarVectorVisionTransformer::new(views.vision, &StarVectorVisionConfig::default())?;

        let input = Tensor::zeros((1, 3, 224, 224), DType::F32, &Device::Cpu)?;
        let out = vision.forward_sequence(&input)?;
        assert_eq!(out.dims3()?, (1, 257, 1024));
        Ok(())
    }
}
