# Model Evaluation

## Held-out rating prediction

The ALS model reads actual Silver ratings and performs a reproducible random split using seed `42`:

- Training fraction: `0.8`
- Training rows: `800,002`
- Test rows: `200,207`
- Valid predictions evaluated: `200,160`
- Cold-start strategy: `drop`

The saved real-data metrics are:

- RMSE: `0.869412`
- MAE: `0.696190`

RMSE penalizes larger prediction errors more heavily because errors are squared before averaging. MAE is the average absolute rating error and is easier to interpret in rating units. Neither metric measures whether the best movies appear at the top of a recommendation list.

## Ranking evaluation

`src/recsys/model.py` includes Spark-native `calculate_ranking_metrics` and `evaluate_ranking` helpers. The protocol is:

1. Train ALS only on the training interactions.
2. Generate candidate recommendations for users appearing in the held-out set.
3. Remove training interactions from those candidates.
4. Rank the remaining candidates and compare the top `K` with held-out user/movie interactions.
5. Report micro Precision@K and macro Recall@K.

The evaluator is covered by a small Spark fixture test. Real-data ranking metrics were not regenerated during release preparation because that would require another model/recommendation computation; no real Precision@K or Recall@K values are claimed here.

## Limitations

- A random interaction split can be optimistic compared with a time-based production split.
- The current evaluation does not model catalog exposure, novelty, diversity, or business constraints.
- ALS predictions can exceed the original rating range; recommendation output is treated as a ranking score, not a guaranteed user rating.
- A new user receives popularity fallback recommendations until sufficient history is available.