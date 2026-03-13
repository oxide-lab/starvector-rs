use std::path::{Path, PathBuf};

use candle::{Device, Tensor};
use image::imageops::FilterType;
use image::{DynamicImage, GenericImage, Rgb, RgbImage};
use thiserror::Error;

use crate::config::MetadataError;
use crate::types::ParsedModelMetadata;

#[derive(Debug, Error)]
pub enum ImagePreprocessorError {
    #[error(transparent)]
    Metadata(#[from] MetadataError),
    #[error("failed to open image {path}: {source}")]
    OpenImage {
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },
    #[error("failed to copy image content into padded canvas")]
    CopyToCanvas,
    #[error("failed to create tensor from preprocessed image: {0}")]
    Tensor(#[from] candle::Error),
}

#[derive(Debug, Clone)]
pub struct ImagePreprocessor {
    size: usize,
    mean: [f32; 3],
    std: [f32; 3],
}

impl ImagePreprocessor {
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self, ImagePreprocessorError> {
        let metadata = ParsedModelMetadata::from_model_dir(model_dir.as_ref())?;
        let mean = metadata.processor_config.mean.map(|v| v as f32);
        let std = metadata.processor_config.std.map(|v| v as f32);

        Ok(Self {
            size: metadata.processor_config.size,
            mean,
            std,
        })
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn mean(&self) -> [f32; 3] {
        self.mean
    }

    pub fn std(&self) -> [f32; 3] {
        self.std
    }

    pub fn preprocess_to_chw_vec(
        &self,
        image_path: impl AsRef<Path>,
    ) -> Result<Vec<f32>, ImagePreprocessorError> {
        let image_path = image_path.as_ref();
        let image =
            image::open(image_path).map_err(|source| ImagePreprocessorError::OpenImage {
                path: image_path.to_path_buf(),
                source,
            })?;

        let rgb = self.normalize_input_mode(image);
        let padded = self.pad_to_square_white(&rgb)?;
        let resized = image::imageops::resize(
            &padded,
            self.size as u32,
            self.size as u32,
            FilterType::CatmullRom,
        );

        Ok(self.rgb_to_normalized_chw(&resized))
    }

    pub fn preprocess_to_tensor(
        &self,
        image_path: impl AsRef<Path>,
        device: &Device,
    ) -> Result<Tensor, ImagePreprocessorError> {
        let vec = self.preprocess_to_chw_vec(image_path)?;
        let tensor = Tensor::from_vec(vec, (3, self.size, self.size), device)?;
        Ok(tensor)
    }

    fn normalize_input_mode(&self, image: DynamicImage) -> RgbImage {
        match image {
            // Match Python reference: RGBA is converted directly to RGB, no alpha compositing.
            DynamicImage::ImageRgba8(img) => DynamicImage::ImageRgba8(img).to_rgb8(),
            DynamicImage::ImageRgba16(img) => DynamicImage::ImageRgba16(img).to_rgb8(),
            DynamicImage::ImageRgb8(img) => img,
            DynamicImage::ImageRgb16(img) => DynamicImage::ImageRgb16(img).to_rgb8(),
            other => other.to_rgb8(),
        }
    }

    fn pad_to_square_white(&self, image: &RgbImage) -> Result<RgbImage, ImagePreprocessorError> {
        let (width, height) = image.dimensions();
        let max_dim = width.max(height);
        let left = (max_dim - width) / 2;
        let top = (max_dim - height) / 2;

        let mut canvas = RgbImage::from_pixel(max_dim, max_dim, Rgb([255, 255, 255]));
        canvas
            .copy_from(image, left, top)
            .map_err(|_| ImagePreprocessorError::CopyToCanvas)?;
        Ok(canvas)
    }

    fn rgb_to_normalized_chw(&self, image: &RgbImage) -> Vec<f32> {
        let (width, height) = image.dimensions();
        let hw = (width as usize) * (height as usize);
        let mut out = vec![0.0_f32; 3 * hw];

        for (i, pixel) in image.pixels().enumerate() {
            let r = (pixel[0] as f32) / 255.0;
            let g = (pixel[1] as f32) / 255.0;
            let b = (pixel[2] as f32) / 255.0;

            out[i] = (r - self.mean[0]) / self.std[0];
            out[hw + i] = (g - self.mean[1]) / self.std[1];
            out[(2 * hw) + i] = (b - self.mean[2]) / self.std[2];
        }

        out
    }
}
