use candle::{Result, Tensor};
use candle_nn::{BatchNorm, Linear, Module, ModuleT, VarBuilder, linear};
use thiserror::Error;

use crate::model::loader::{LoaderError, ReuseFirstWeightLoader};

const BN_TRACKED_KEY: &str = "model.image_projection.norm.num_batches_tracked";

#[derive(Debug, Error)]
pub enum AdapterError {
    #[error(transparent)]
    Candle(#[from] candle::Error),
    #[error(transparent)]
    Loader(#[from] LoaderError),
}

#[derive(Debug, Clone)]
pub struct StarVectorAdapter {
    c_fc: Linear,
    c_proj: Linear,
    norm: BatchNorm,
    num_batches_tracked: i64,
}

impl StarVectorAdapter {
    pub fn new(
        vb: VarBuilder,
        loader: &ReuseFirstWeightLoader,
    ) -> std::result::Result<Self, AdapterError> {
        let c_fc = linear(1024, 2048, vb.pp("c_fc"))?;
        let c_proj = linear(2048, 2048, vb.pp("c_proj"))?;

        let running_mean = vb.get(257, "norm.running_mean")?;
        let running_var = vb.get(257, "norm.running_var")?;
        let weight = vb.get(257, "norm.weight")?;
        let bias = vb.get(257, "norm.bias")?;
        let norm = BatchNorm::new(257, running_mean, running_var, weight, bias, 1e-5)?;

        let num_batches_tracked = loader.load_i64_scalar_non_float(BN_TRACKED_KEY)?;

        Ok(Self {
            c_fc,
            c_proj,
            norm,
            num_batches_tracked,
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
