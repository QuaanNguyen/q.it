use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalObservation {
    pub schema_version: u32,
    pub method: String,
    pub query: Vec<Vec<f64>>,
    pub documents: Vec<Vec<Vec<f64>>>,
    pub scores: Vec<f64>,
    pub ranking: Vec<usize>,
    pub tie_break: String,
}

impl RetrievalObservation {
    pub fn validate(&self, document_count: usize) -> Result<(), String> {
        if self.schema_version != 1 || self.method != "maxsim" || self.tie_break != "document_order"
        {
            return Err("unsupported late-interaction retrieval contract".into());
        }
        if self.query.is_empty()
            || self.documents.len() != document_count
            || self.scores.len() != document_count
            || self.ranking.len() != document_count
            || document_count == 0
        {
            return Err("retrieval must retain the query and every candidate document".into());
        }
        let mut scalar_count = 0usize;
        for vectors in std::iter::once(&self.query).chain(self.documents.iter()) {
            if vectors.is_empty() || vectors.len() > 4096 {
                return Err("retrieval matrices require between one and 4096 token vectors".into());
            }
            for vector in vectors {
                scalar_count += vector.len();
                if vector.len() != 128 || vector.iter().any(|value| !value.is_finite()) {
                    return Err(
                        "retrieval requires finite native 128-dimensional vectors".into(),
                    );
                }
            }
        }
        if scalar_count > 1_000_000 {
            return Err("raw retrieval observations exceed the one-million-scalar storage limit; use a smaller versioned retrieval case".into());
        }
        for (document, observed) in self.documents.iter().zip(&self.scores) {
            let expected = self
                .query
                .iter()
                .map(|query| {
                    document
                        .iter()
                        .map(|token| {
                            query
                                .iter()
                                .zip(token)
                                .map(|(left, right)| left * right)
                                .sum::<f64>()
                        })
                        .fold(f64::NEG_INFINITY, f64::max)
                })
                .sum::<f64>();
            if !observed.is_finite()
                || (expected - observed).abs() > 0.0001 * expected.abs().max(1.0)
            {
                return Err(
                    "native retrieval score differs from MaxSim over the retained vectors".into(),
                );
            }
        }
        let mut ranking = (0..document_count).collect::<Vec<_>>();
        ranking.sort_by(|left, right| {
            self.scores[*right]
                .total_cmp(&self.scores[*left])
                .then_with(|| left.cmp(right))
        });
        if self.ranking != ranking {
            return Err(
                "retrieval ranking must match native scores with document-order ties".into(),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_maxsim_preserves_every_vector_and_rejects_inconsistent_results() {
        let mut a = vec![0.0; 128];
        a[0] = 1.0;
        let mut b = vec![0.0; 128];
        b[1] = 1.0;
        let mut result = RetrievalObservation {
            schema_version: 1,
            method: "maxsim".into(),
            query: vec![a.clone(), b.clone()],
            documents: vec![vec![a.clone(), b], vec![a]],
            scores: vec![2.0, 1.0],
            ranking: vec![0, 1],
            tie_break: "document_order".into(),
        };
        result.validate(2).unwrap();
        result.scores[0] = 1.0;
        assert!(result.validate(2).is_err());
        result.scores[0] = 2.0;
        result.documents[0][0].pop();
        assert!(result.validate(2).is_err());
    }
}
