use crate::query_classifier::QueryType;

#[derive(Debug, Clone)]
pub struct MetricScores {
    pub bm25: f64,
    pub cosine: f64,
    pub path_match: f64,
    pub symbol_match: f64,
    pub import_graph: f64,
    pub git_recency: f64,
}

#[derive(Debug, Clone)]
pub struct RankedCandidate {
    pub id: i64,
    pub scores: MetricScores,
    pub rank: usize,
}

const METRIC_NAMES: [&str; 6] = [
    "bm25",
    "cosine",
    "path_match",
    "symbol_match",
    "import_graph",
    "git_recency",
];

struct ColumnWeights {
    bm25: usize,
    cosine: usize,
    path_match: usize,
    symbol_match: usize,
    import_graph: usize,
    git_recency: usize,
}

fn column_weights(query_type: &QueryType) -> ColumnWeights {
    match query_type {
        QueryType::Identifier => ColumnWeights {
            bm25: 2,
            cosine: 1,
            path_match: 1,
            symbol_match: 2,
            import_graph: 1,
            git_recency: 1,
        },
        QueryType::NaturalLanguage => ColumnWeights {
            bm25: 1,
            cosine: 2,
            path_match: 1,
            symbol_match: 1,
            import_graph: 1,
            git_recency: 1,
        },
        QueryType::PathLike => ColumnWeights {
            bm25: 1,
            cosine: 1,
            path_match: 3,
            symbol_match: 1,
            import_graph: 1,
            git_recency: 1,
        },
    }
}

const EPSILON: f64 = 0.05;

fn get_score(scores: &MetricScores, metric: &str) -> f64 {
    match metric {
        "bm25" => scores.bm25,
        "cosine" => scores.cosine,
        "path_match" => scores.path_match,
        "symbol_match" => scores.symbol_match,
        "import_graph" => scores.import_graph,
        "git_recency" => scores.git_recency,
        _ => 0.0,
    }
}

fn get_weight(weights: &ColumnWeights, metric: &str) -> usize {
    match metric {
        "bm25" => weights.bm25,
        "cosine" => weights.cosine,
        "path_match" => weights.path_match,
        "symbol_match" => weights.symbol_match,
        "import_graph" => weights.import_graph,
        "git_recency" => weights.git_recency,
        _ => 0,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MetricAvailability {
    pub bm25: bool,
    pub cosine: bool,
    pub path_match: bool,
    pub symbol_match: bool,
    pub import_graph: bool,
    pub git_recency: bool,
}

impl MetricAvailability {
    pub fn all_active() -> Self {
        Self {
            bm25: true,
            cosine: true,
            path_match: true,
            symbol_match: true,
            import_graph: true,
            git_recency: true,
        }
    }

    pub fn is_active(&self, metric: &str) -> bool {
        match metric {
            "bm25" => self.bm25,
            "cosine" => self.cosine,
            "path_match" => self.path_match,
            "symbol_match" => self.symbol_match,
            "import_graph" => self.import_graph,
            "git_recency" => self.git_recency,
            _ => false,
        }
    }
}

pub fn poem_rank(
    candidates: &std::collections::HashMap<i64, MetricScores>,
    query_type: &QueryType,
    top_k: usize,
    availability: &MetricAvailability,
) -> Vec<RankedCandidate> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let surviving = prune_top_k(candidates, top_k, availability);

    let ids: Vec<i64> = surviving.iter().map(|&(id, _)| id).collect();
    let scores: Vec<&MetricScores> = surviving.iter().map(|&(_, s)| s).collect();

    if ids.len() == 1 {
        return vec![RankedCandidate {
            id: ids[0],
            scores: scores[0].clone(),
            rank: 0,
        }];
    }

    let weights = column_weights(query_type);
    let n = ids.len();
    let total_weight = METRIC_NAMES
        .iter()
        .filter(|m| availability.is_active(m))
        .map(|m| get_weight(&weights, m))
        .sum::<usize>() as f64;

    if total_weight == 0.0 {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| ids[a].cmp(&ids[b]));
        return order
            .into_iter()
            .enumerate()
            .map(|(rank, i)| RankedCandidate {
                id: ids[i],
                scores: scores[i].clone(),
                rank,
            })
            .collect();
    }

    let mut counts = vec![0u16; n * n];

    for metric in &METRIC_NAMES {
        let weight = get_weight(&weights, metric) as u16;
        if weight == 0 || !availability.is_active(metric) {
            continue;
        }

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            let sa = get_score(scores[a], metric);
            let sb = get_score(scores[b], metric);
            sb.partial_cmp(&sa)
                .expect("floats should be comparable")
                .then_with(|| ids[a].cmp(&ids[b]))
        });

        let k = top_k.min(n);

        for ri in 0..k {
            let i = indices[ri];
            for rj in (ri + 1)..k {
                counts[i * n + indices[rj]] += weight;
            }
        }
    }

    let threshold = total_weight * 0.5;
    let mut fitness = vec![0.0_f64; n];

    for i in 0..n {
        let mut sum_dom = 0.0_f64;
        let mut num_dominating = 0usize;
        let mut num_submitting = 0usize;

        for j in 0..n {
            if i == j {
                continue;
            }
            let count = counts[i * n + j] as f64;
            sum_dom += count;
            if count > threshold {
                num_dominating += 1;
            }
            if count < threshold {
                num_submitting += 1;
            }
        }

        let mean_dom = if n > 1 {
            sum_dom / ((n - 1) as f64 * total_weight)
        } else {
            0.0
        };
        fitness[i] =
            mean_dom * (num_dominating as f64 + EPSILON) / (num_submitting as f64 + EPSILON);
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        fitness[b]
            .partial_cmp(&fitness[a])
            .expect("floats should be comparable")
            .then_with(|| ids[a].cmp(&ids[b]))
    });

    order
        .into_iter()
        .enumerate()
        .map(|(rank, idx)| RankedCandidate {
            id: ids[idx],
            scores: scores[idx].clone(),
            rank,
        })
        .collect()
}

fn prune_top_k<'a>(
    candidates: &'a std::collections::HashMap<i64, MetricScores>,
    top_k: usize,
    availability: &MetricAvailability,
) -> Vec<(i64, &'a MetricScores)> {
    if candidates.len() <= top_k {
        return candidates.iter().map(|(&id, s)| (id, s)).collect();
    }

    let mut surviving = std::collections::HashSet::new();

    for metric in &METRIC_NAMES {
        if !availability.is_active(metric) {
            continue;
        }
        let mut pairs: Vec<(i64, f64)> = candidates
            .iter()
            .map(|(&id, s)| (id, get_score(s, metric)))
            .collect();
        pairs.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .expect("floats")
                .then_with(|| a.0.cmp(&b.0))
        });
        for (id, _) in pairs.into_iter().take(top_k) {
            surviving.insert(id);
        }
    }

    candidates
        .iter()
        .filter(|(id, _)| surviving.contains(id))
        .map(|(&id, s)| (id, s))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_candidates_returns_empty() {
        let candidates = std::collections::HashMap::new();
        let result = poem_rank(
            &candidates,
            &QueryType::Identifier,
            1000,
            &MetricAvailability::all_active(),
        );
        assert!(result.is_empty());
    }

    #[test]
    fn single_candidate_gets_rank_zero() {
        let mut candidates = std::collections::HashMap::new();
        candidates.insert(
            1,
            MetricScores {
                bm25: 0.5,
                cosine: 0.8,
                path_match: 0.0,
                symbol_match: 0.3,
                import_graph: 0.0,
                git_recency: 0.5,
            },
        );
        let result = poem_rank(
            &candidates,
            &QueryType::Identifier,
            1000,
            &MetricAvailability::all_active(),
        );
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].rank, 0);
        assert_eq!(result[0].id, 1);
    }

    #[test]
    fn higher_scoring_candidate_ranks_better() {
        let mut candidates = std::collections::HashMap::new();
        candidates.insert(
            1,
            MetricScores {
                bm25: 0.9,
                cosine: 0.9,
                path_match: 0.9,
                symbol_match: 0.9,
                import_graph: 0.5,
                git_recency: 0.5,
            },
        );
        candidates.insert(
            2,
            MetricScores {
                bm25: 0.1,
                cosine: 0.1,
                path_match: 0.1,
                symbol_match: 0.1,
                import_graph: 0.5,
                git_recency: 0.5,
            },
        );
        let result = poem_rank(
            &candidates,
            &QueryType::NaturalLanguage,
            1000,
            &MetricAvailability::all_active(),
        );
        assert_eq!(
            result[0].id, 1,
            "Higher-scoring candidate should rank first"
        );
        assert_eq!(result[1].id, 2);
    }

    #[test]
    fn deterministic_ranking_for_same_inputs() {
        let mut candidates = std::collections::HashMap::new();
        candidates.insert(
            1,
            MetricScores {
                bm25: 0.5,
                cosine: 0.7,
                path_match: 0.3,
                symbol_match: 0.4,
                import_graph: 0.2,
                git_recency: 0.6,
            },
        );
        candidates.insert(
            2,
            MetricScores {
                bm25: 0.3,
                cosine: 0.5,
                path_match: 0.7,
                symbol_match: 0.2,
                import_graph: 0.8,
                git_recency: 0.4,
            },
        );
        candidates.insert(
            3,
            MetricScores {
                bm25: 0.7,
                cosine: 0.3,
                path_match: 0.5,
                symbol_match: 0.6,
                import_graph: 0.1,
                git_recency: 0.9,
            },
        );

        let result1 = poem_rank(
            &candidates,
            &QueryType::Identifier,
            1000,
            &MetricAvailability::all_active(),
        );
        let result2 = poem_rank(
            &candidates,
            &QueryType::Identifier,
            1000,
            &MetricAvailability::all_active(),
        );

        for (a, b) in result1.iter().zip(result2.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.rank, b.rank);
        }
    }

    #[test]
    fn masked_metric_excluded_from_tournament() {
        let cands = vec![
            (
                0,
                MetricScores {
                    bm25: 0.2,
                    cosine: 0.2,
                    path_match: 1.0,
                    symbol_match: 0.0,
                    import_graph: 0.0,
                    git_recency: 1.0,
                },
            ),
            (
                1,
                MetricScores {
                    bm25: 0.5,
                    cosine: 1.0,
                    path_match: 0.5,
                    symbol_match: 0.5,
                    import_graph: 1.0,
                    git_recency: 0.5,
                },
            ),
            (
                2,
                MetricScores {
                    bm25: 0.5,
                    cosine: 1.0,
                    path_match: 0.0,
                    symbol_match: 0.0,
                    import_graph: 1.0,
                    git_recency: 0.5,
                },
            ),
            (
                3,
                MetricScores {
                    bm25: 0.2,
                    cosine: 0.5,
                    path_match: 0.0,
                    symbol_match: 0.5,
                    import_graph: 0.5,
                    git_recency: 0.0,
                },
            ),
        ];
        let mut candidates = std::collections::HashMap::new();
        for (id, s) in cands {
            candidates.insert(id, s);
        }

        let active = poem_rank(
            &candidates,
            &QueryType::Identifier,
            1000,
            &MetricAvailability::all_active(),
        );
        let masked = {
            let mut a = MetricAvailability::all_active();
            a.cosine = false;
            poem_rank(&candidates, &QueryType::Identifier, 1000, &a)
        };

        let ids_active: Vec<i64> = active.iter().map(|r| r.id).collect();
        let ids_masked: Vec<i64> = masked.iter().map(|r| r.id).collect();
        assert_eq!(ids_active, vec![1, 2, 0, 3]);
        assert_eq!(ids_masked, vec![1, 0, 2, 3]);
        assert_ne!(
            ids_active, ids_masked,
            "masking cosine should change ranking"
        );
    }

    #[test]
    fn deterministic_tie_break_by_id() {
        let mut candidates = std::collections::HashMap::new();
        let s = MetricScores {
            bm25: 0.5,
            cosine: 0.5,
            path_match: 0.5,
            symbol_match: 0.5,
            import_graph: 0.5,
            git_recency: 0.5,
        };
        candidates.insert(3, s.clone());
        candidates.insert(1, s.clone());
        candidates.insert(2, s.clone());

        for _ in 0..10 {
            let result = poem_rank(
                &candidates,
                &QueryType::Identifier,
                1000,
                &MetricAvailability::all_active(),
            );
            let ids: Vec<i64> = result.iter().map(|r| r.id).collect();
            assert_eq!(ids, vec![1, 2, 3], "lower ID should rank first on ties");
        }
    }

    #[test]
    fn deterministic_with_shuffled_insertion_order() {
        let entries: Vec<(i64, MetricScores)> = vec![
            (
                1,
                MetricScores {
                    bm25: 0.5,
                    cosine: 0.7,
                    path_match: 0.3,
                    symbol_match: 0.4,
                    import_graph: 0.2,
                    git_recency: 0.6,
                },
            ),
            (
                2,
                MetricScores {
                    bm25: 0.3,
                    cosine: 0.5,
                    path_match: 0.7,
                    symbol_match: 0.2,
                    import_graph: 0.8,
                    git_recency: 0.4,
                },
            ),
            (
                3,
                MetricScores {
                    bm25: 0.7,
                    cosine: 0.3,
                    path_match: 0.5,
                    symbol_match: 0.6,
                    import_graph: 0.1,
                    git_recency: 0.9,
                },
            ),
        ];

        let mk = |entries: &[(i64, MetricScores)]| {
            let mut m = std::collections::HashMap::new();
            for (id, s) in entries {
                m.insert(*id, s.clone());
            }
            m
        };

        let m1 = mk(&entries);
        // Rebuild in a different insertion order.
        let order: Vec<usize> = vec![2, 0, 1];
        let mut m2 = std::collections::HashMap::new();
        for &i in &order {
            m2.insert(entries[i].0, entries[i].1.clone());
        }

        let r1 = poem_rank(
            &m1,
            &QueryType::Identifier,
            1000,
            &MetricAvailability::all_active(),
        );
        let r2 = poem_rank(
            &m2,
            &QueryType::Identifier,
            1000,
            &MetricAvailability::all_active(),
        );

        let ids1: Vec<i64> = r1.iter().map(|r| r.id).collect();
        let ids2: Vec<i64> = r2.iter().map(|r| r.id).collect();
        assert_eq!(ids1, ids2);
    }

    #[test]
    fn dynamic_total_weight_with_masked_metrics() {
        let mut candidates = std::collections::HashMap::new();
        candidates.insert(
            1,
            MetricScores {
                bm25: 0.9,
                cosine: 0.1,
                path_match: 0.5,
                symbol_match: 0.5,
                import_graph: 0.5,
                git_recency: 0.5,
            },
        );
        candidates.insert(
            2,
            MetricScores {
                bm25: 0.5,
                cosine: 0.1,
                path_match: 0.5,
                symbol_match: 0.5,
                import_graph: 0.5,
                git_recency: 0.5,
            },
        );

        let full = MetricAvailability::all_active();
        let masked = {
            let mut a = MetricAvailability::all_active();
            a.cosine = false;
            a
        };

        let r_full = poem_rank(&candidates, &QueryType::Identifier, 1000, &full);
        let r_masked = poem_rank(&candidates, &QueryType::Identifier, 1000, &masked);

        // With cosine active, candidate 1's cosine loss is weighted 1, keeping
        // the bm25 gap dominant; both should still rank 1 first.
        assert_eq!(r_full[0].id, 1);
        assert_eq!(r_masked[0].id, 1);
    }

    #[test]
    fn all_metrics_inactive_returns_sorted_by_id() {
        let mut candidates = std::collections::HashMap::new();
        let s = MetricScores {
            bm25: 0.5,
            cosine: 0.5,
            path_match: 0.5,
            symbol_match: 0.5,
            import_graph: 0.5,
            git_recency: 0.5,
        };
        candidates.insert(3, s.clone());
        candidates.insert(1, s.clone());
        candidates.insert(2, s.clone());

        let inactive = MetricAvailability {
            bm25: false,
            cosine: false,
            path_match: false,
            symbol_match: false,
            import_graph: false,
            git_recency: false,
        };

        let result = poem_rank(&candidates, &QueryType::Identifier, 1000, &inactive);
        let ids: Vec<i64> = result.iter().map(|r| r.id).collect();
        assert_eq!(
            ids,
            vec![1, 2, 3],
            "all-inactive should return sorted by ID"
        );
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn prune_top_k_deterministic_with_dead_metric() {
        let mut candidates = std::collections::HashMap::new();
        for id in 0..1001i64 {
            candidates.insert(
                id,
                MetricScores {
                    bm25: if id == 500 { 1.0 } else { 0.0 },
                    cosine: 0.0,
                    path_match: 0.0,
                    symbol_match: 0.0,
                    import_graph: 0.0,
                    git_recency: 0.0,
                },
            );
        }

        let mut masked = MetricAvailability::all_active();
        masked.cosine = false;
        masked.path_match = false;
        masked.symbol_match = false;
        masked.import_graph = false;
        masked.git_recency = false;

        let result = poem_rank(&candidates, &QueryType::Identifier, 1000, &masked);
        let ids: Vec<i64> = result.iter().map(|r| r.id).collect();
        // bm25 picks id 500 first, then the remaining 1000 tie at 0.0 and the
        // ID tie-breaker admits the 999 lowest IDs. So exactly 1000 survive.
        assert_eq!(ids.len(), 1000);
        assert_eq!(ids[0], 500);
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert!(sorted.windows(2).all(|w| w[0] != w[1]), "no duplicates");
    }
}
