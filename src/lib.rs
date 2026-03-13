use std::str::FromStr;

use thiserror::Error;

pub mod config;
pub mod model;
pub mod types;
pub use model::generation::{GenerationConfig, GenerationOutput};
pub use model::starvector::{PrecisionPolicy, StarVector};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeDevice {
    Cpu,
    Cuda(usize),
}

impl RuntimeDevice {
    pub fn ensure_supported(&self) -> Result<(), StarVectorError> {
        match self {
            RuntimeDevice::Cpu => Ok(()),
            RuntimeDevice::Cuda(_) => {
                if cfg!(feature = "cuda") {
                    Ok(())
                } else {
                    Err(StarVectorError::CudaNotEnabled)
                }
            }
        }
    }

    pub fn to_candle_device(&self) -> Result<candle::Device, StarVectorError> {
        self.ensure_supported()?;
        match self {
            RuntimeDevice::Cpu => Ok(candle::Device::Cpu),
            RuntimeDevice::Cuda(index) => cuda_device(*index),
        }
    }
}

impl FromStr for RuntimeDevice {
    type Err = StarVectorError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq_ignore_ascii_case("cpu") {
            return Ok(Self::Cpu);
        }

        let lower = s.to_ascii_lowercase();
        if lower == "cuda" {
            return Ok(Self::Cuda(0));
        }
        if let Some(idx) = lower.strip_prefix("cuda:") {
            let index = idx
                .parse::<usize>()
                .map_err(|_| StarVectorError::InvalidDevice {
                    value: s.to_owned(),
                })?;
            return Ok(Self::Cuda(index));
        }

        Err(StarVectorError::InvalidDevice {
            value: s.to_owned(),
        })
    }
}

#[derive(Debug, Error)]
pub enum StarVectorError {
    #[error("invalid device '{value}', expected 'cpu', 'cuda', or 'cuda:<index>'")]
    InvalidDevice { value: String },

    #[error(
        "binary built without cuda feature; rebuild with `cargo build --features cuda` to use --device cuda[:idx]"
    )]
    CudaNotEnabled,

    #[error("failed to initialize cuda device cuda:{index}: {source}")]
    CudaInit {
        index: usize,
        #[source]
        source: candle::Error,
    },
}

#[cfg(feature = "cuda")]
fn cuda_device(index: usize) -> Result<candle::Device, StarVectorError> {
    candle::Device::new_cuda(index).map_err(|source| StarVectorError::CudaInit { index, source })
}

#[cfg(not(feature = "cuda"))]
fn cuda_device(_index: usize) -> Result<candle::Device, StarVectorError> {
    Err(StarVectorError::CudaNotEnabled)
}
