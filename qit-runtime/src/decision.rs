use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const PROBABILITY_SUM_TOLERANCE: f64 = 1e-6;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionInput {
    Text,
    Image,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionCandidate {
    pub id: String,
    pub text: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionCase {
    pub state: String,
    pub question: String,
    pub candidates: Vec<DecisionCandidate>,
    #[serde(default)]
    pub expected: Option<String>,
}

impl DecisionCase {
    pub fn validate(&self) -> Result<(), String> {
        if self.state.trim().is_empty() || self.question.trim().is_empty() {
            return Err("decision state and question must be non-empty".into());
        }
        if !(2..=10).contains(&self.candidates.len()) {
            return Err("a decision requires two to ten ordered candidates".into());
        }
        let mut ids = HashSet::new();
        for candidate in &self.candidates {
            if candidate.id.is_empty()
                || candidate.id.len() > 80
                || !candidate
                    .id
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
                || candidate.text.trim().is_empty()
                || !ids.insert(candidate.id.as_str())
            {
                return Err(
                    "decision candidates require unique portable identifiers and non-empty text"
                        .into(),
                );
            }
        }
        if self
            .expected
            .as_deref()
            .is_some_and(|expected| !ids.contains(expected))
        {
            return Err("decision expected answer must identify a candidate".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionConfiguration {
    pub schema_version: u32,
    pub inputs: Vec<DecisionInput>,
    pub calibration: String,
    pub selection_rule: String,
    pub preprocessing: String,
}

impl DecisionConfiguration {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1
            || self.inputs.is_empty()
            || self.inputs.len() > 2
            || (self.inputs.len() == 2 && self.inputs[0] == self.inputs[1])
            || self.calibration.trim().is_empty()
            || self.selection_rule.trim().is_empty()
            || self.preprocessing.trim().is_empty()
        {
            return Err("decision configuration requires protocol version 1, distinct input types, calibration, selection rule, and preprocessing identity".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateProbability {
    pub id: String,
    pub probability: f64,
    #[serde(default)]
    pub logit: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionProvenance {
    pub calibration: String,
    pub selection_rule: String,
    pub preprocessing: String,
    pub input_digest: String,
    pub input_truncated: bool,
    pub native: Value,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionResult {
    pub schema_version: u32,
    pub candidates: Vec<CandidateProbability>,
    pub selected_id: String,
    pub provenance: DecisionProvenance,
}

impl DecisionResult {
    pub fn validate(
        &self,
        case: &DecisionCase,
        configuration: &DecisionConfiguration,
    ) -> Result<(), String> {
        if self.schema_version != 1 || self.candidates.len() != case.candidates.len() {
            return Err(
                "decision result must use protocol version 1 and include every candidate".into(),
            );
        }
        for (returned, expected) in self.candidates.iter().zip(&case.candidates) {
            if returned.id != expected.id {
                return Err("decision result must preserve candidate identity and order".into());
            }
            if !returned.probability.is_finite()
                || !(0.0..=1.0).contains(&returned.probability)
                || returned.logit.is_some_and(|value| !value.is_finite())
            {
                return Err("decision probabilities must be finite values between zero and one, and supplied logits must be finite".into());
            }
        }
        let sum: f64 = self
            .candidates
            .iter()
            .map(|candidate| candidate.probability)
            .sum();
        if (sum - 1.0).abs() > PROBABILITY_SUM_TOLERANCE {
            return Err("decision probabilities must sum to one within tolerance 0.000001; results are not renormalized".into());
        }
        let selected = self
            .candidates
            .iter()
            .find(|candidate| candidate.id == self.selected_id)
            .ok_or_else(|| "decision selected answer does not identify a candidate".to_string())?;
        if self
            .candidates
            .iter()
            .any(|candidate| candidate.probability > selected.probability)
        {
            return Err("decision selected answer must have maximal native probability".into());
        }
        let provenance = &self.provenance;
        if provenance.calibration != configuration.calibration
            || provenance.selection_rule != configuration.selection_rule
            || provenance.preprocessing != configuration.preprocessing
        {
            return Err(
                "decision result provenance differs from the recorded model configuration".into(),
            );
        }
        if provenance.input_truncated {
            return Err(
                "decision input was truncated; full-input comparisons require all case content"
                    .into(),
            );
        }
        if provenance.input_digest.len() != 64
            || !provenance
                .input_digest
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
            || !provenance.native.is_object()
        {
            return Err("decision provenance requires a SHA-256 rendered-input digest and native evidence object".into());
        }
        Ok(())
    }

    pub fn score(&self, case: &DecisionCase) -> Option<DecisionMetrics> {
        let expected = case.expected.as_deref()?;
        let gold = self
            .candidates
            .iter()
            .find(|candidate| candidate.id == expected)?;
        Some(DecisionMetrics {
            brier_score: self
                .candidates
                .iter()
                .map(|candidate| {
                    let target = if candidate.id == expected { 1.0 } else { 0.0 };
                    (candidate.probability - target).powi(2)
                })
                .sum(),
            negative_log_likelihood: if gold.probability == 0.0 {
                LogLoss::Infinite
            } else {
                LogLoss::Finite(-gold.probability.ln())
            },
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "status", content = "value", rename_all = "snake_case")]
pub enum LogLoss {
    Finite(f64),
    Infinite,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DecisionMetrics {
    pub brier_score: f64,
    pub negative_log_likelihood: LogLoss,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DecisionObservation {
    pub case: DecisionCase,
    pub media: Option<String>,
    pub result: DecisionResult,
    pub metrics: Option<DecisionMetrics>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DecisionSummary {
    pub scored_sample_count: u32,
    pub brier_score_mean: f64,
    pub negative_log_likelihood: LogLoss,
    pub zero_gold_probability_count: u32,
}

pub fn summarize<'a>(
    metrics: impl Iterator<Item = &'a DecisionMetrics>,
) -> Option<DecisionSummary> {
    let metrics = metrics.collect::<Vec<_>>();
    if metrics.is_empty() {
        return None;
    }
    let count = metrics.len() as f64;
    let zero_gold_probability_count = metrics
        .iter()
        .filter(|metric| matches!(metric.negative_log_likelihood, LogLoss::Infinite))
        .count() as u32;
    Some(DecisionSummary {
        scored_sample_count: metrics.len() as u32,
        brier_score_mean: metrics
            .iter()
            .map(|metric| metric.brier_score / count)
            .sum(),
        negative_log_likelihood: if zero_gold_probability_count > 0 {
            LogLoss::Infinite
        } else {
            LogLoss::Finite(
                metrics
                    .iter()
                    .map(|metric| match metric.negative_log_likelihood {
                        LogLoss::Finite(value) => value / count,
                        LogLoss::Infinite => unreachable!(),
                    })
                    .sum(),
            )
        },
        zero_gold_probability_count,
    })
}
