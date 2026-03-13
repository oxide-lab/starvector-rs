use candle::{D, IndexOp, Result, Tensor};
use candle_transformers::generation::LogitsProcessor;
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub do_sample: bool,
    pub temperature: f64,
    pub top_p: f64,
    pub repetition_penalty: f64,
    pub seed: u64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 7800,
            do_sample: false,
            temperature: 0.2,
            top_p: 0.95,
            repetition_penalty: 1.0,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationOutput {
    pub svg: String,
    pub token_ids: Vec<u32>,
}

pub fn has_suffix(haystack: &[u32], suffix: &[u32]) -> bool {
    if suffix.is_empty() {
        return true;
    }
    haystack.ends_with(suffix)
}

pub fn greedy_next_token_id(logits: &Tensor) -> Result<u32> {
    let argmax = logits.argmax(D::Minus1)?;
    if let Ok(ids) = argmax.to_vec1::<u32>() {
        return ids
            .first()
            .copied()
            .ok_or_else(|| candle::Error::msg("argmax returned empty tensor"));
    }
    let ids = argmax.to_vec1::<i64>()?;
    let value = ids
        .first()
        .copied()
        .ok_or_else(|| candle::Error::msg("argmax returned empty tensor"))?;
    u32::try_from(value).map_err(|_| candle::Error::msg("argmax id does not fit into u32"))
}

pub fn next_token_id(
    logits: &Tensor,
    history_token_ids: &[u32],
    cfg: &GenerationConfig,
    logits_processor: &mut LogitsProcessor,
) -> Result<u32> {
    let logits_1d = logits.i(0)?;
    let processed_logits =
        apply_repetition_penalty(&logits_1d, history_token_ids, cfg.repetition_penalty)?;
    if cfg.do_sample {
        logits_processor.sample(&processed_logits)
    } else {
        let logits_2d = processed_logits.unsqueeze(0)?;
        greedy_next_token_id(&logits_2d)
    }
}

fn apply_repetition_penalty(
    logits: &Tensor,
    history_token_ids: &[u32],
    repetition_penalty: f64,
) -> Result<Tensor> {
    if repetition_penalty <= 1.0 || history_token_ids.is_empty() {
        return Ok(logits.clone());
    }

    let mut data = logits.to_vec1::<f32>()?;
    let mut seen = HashSet::with_capacity(history_token_ids.len());
    for &token in history_token_ids {
        let idx = token as usize;
        if idx >= data.len() || !seen.insert(idx) {
            continue;
        }
        if data[idx] < 0.0 {
            data[idx] *= repetition_penalty as f32;
        } else {
            data[idx] /= repetition_penalty as f32;
        }
    }

    Tensor::from_slice(&data, data.len(), logits.device())
}

#[cfg(test)]
mod tests {
    use candle::{Device, Tensor};

    use super::{apply_repetition_penalty, greedy_next_token_id, has_suffix};

    #[test]
    fn suffix_detection_matches_expected_behavior() {
        assert!(has_suffix(&[1, 2, 3], &[2, 3]));
        assert!(has_suffix(&[1, 2, 3], &[]));
        assert!(!has_suffix(&[1, 2, 3], &[1, 3]));
        assert!(!has_suffix(&[1, 2], &[1, 2, 3]));
    }

    #[test]
    fn greedy_step_picks_highest_logit() -> candle::Result<()> {
        let logits = Tensor::from_slice(&[0.1_f32, 3.0, -0.5], (1, 3), &Device::Cpu)?;
        let next = greedy_next_token_id(&logits)?;
        assert_eq!(next, 1);
        Ok(())
    }

    #[test]
    fn repetition_penalty_demotes_seen_tokens() -> candle::Result<()> {
        let logits = Tensor::from_slice(&[4.0_f32, 3.0, -2.0], 3, &Device::Cpu)?;
        let penalized = apply_repetition_penalty(&logits, &[0, 2, 2], 2.0)?;
        let v = penalized.to_vec1::<f32>()?;
        assert!((v[0] - 2.0).abs() < 1e-6);
        assert!((v[1] - 3.0).abs() < 1e-6);
        assert!((v[2] + 4.0).abs() < 1e-6);
        Ok(())
    }
}
