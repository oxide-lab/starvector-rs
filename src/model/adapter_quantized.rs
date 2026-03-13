use std::collections::HashMap;
use std::sync::Arc;

use candle::quantized::{QMatMul, QTensor, gguf_file};
use candle::{DType, Result, Tensor};
use candle_nn::{BatchNorm, Module, ModuleT};

#[derive(Clone)]
struct QuantStore {
    tensors: Arc<HashMap<String, Arc<QTensor>>>,
    device: candle::Device,
}

impl QuantStore {
    fn from_gguf(path: &std::path::Path, device: &candle::Device) -> Result<Self> {
        const ROOT: &str = "model.image_projection.";
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

#[derive(Debug, Clone)]
pub struct StarVectorAdapterQuantized {
    c_fc: QuantLinear,
    c_proj: QuantLinear,
    norm: BatchNorm,
    num_batches_tracked: i64,
}

impl StarVectorAdapterQuantized {
    pub fn load_from_gguf(path: &std::path::Path, device: &candle::Device) -> Result<Self> {
        let store = QuantStore::from_gguf(path, device)?;
        let c_fc_w = store.get_with_shape("model.image_projection.c_fc.weight", &[2048, 1024])?;
        let c_fc_b = store
            .get_with_shape("model.image_projection.c_fc.bias", &[2048])?
            .dequantize(&store.device)?;
        let c_fc = QuantLinear::from_qtensor(c_fc_w, Some(c_fc_b))?;

        let c_proj_w =
            store.get_with_shape("model.image_projection.c_proj.weight", &[2048, 2048])?;
        let c_proj_b = store
            .get_with_shape("model.image_projection.c_proj.bias", &[2048])?
            .dequantize(&store.device)?;
        let c_proj = QuantLinear::from_qtensor(c_proj_w, Some(c_proj_b))?;

        let running_mean = store
            .get_with_shape("model.image_projection.norm.running_mean", &[257])?
            .dequantize(&store.device)?;
        let running_var = store
            .get_with_shape("model.image_projection.norm.running_var", &[257])?
            .dequantize(&store.device)?;
        let weight = store
            .get_with_shape("model.image_projection.norm.weight", &[257])?
            .dequantize(&store.device)?;
        let bias = store
            .get_with_shape("model.image_projection.norm.bias", &[257])?
            .dequantize(&store.device)?;
        let (running_mean, running_var, weight, bias) = if store.device.is_cuda() {
            (
                running_mean.to_dtype(candle::DType::F16)?,
                running_var.to_dtype(candle::DType::F16)?,
                weight.to_dtype(candle::DType::F16)?,
                bias.to_dtype(candle::DType::F16)?,
            )
        } else {
            (running_mean, running_var, weight, bias)
        };
        let norm = BatchNorm::new(257, running_mean, running_var, weight, bias, 1e-5)?;

        Ok(Self {
            c_fc,
            c_proj,
            norm,
            num_batches_tracked: 0,
        })
    }

    pub fn num_batches_tracked(&self) -> i64 {
        self.num_batches_tracked
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.c_fc.forward(xs)?;
        let xs = swish(&xs)?;
        let xs = self.c_proj.forward(&xs)?;
        self.norm.forward_t(&xs, false)
    }
}

fn swish(xs: &Tensor) -> Result<Tensor> {
    xs * candle_nn::ops::sigmoid(xs)?
}
