# Model Card

- Benchmark models: right-censored Gaussian baselines (global intercept,
  scaffold, hierarchy, motif, one-hot k-mer ridge, position-additive, edit KNN,
  mutation graph, small MLP, corrected_v1_31).
- Candidate: support_aware_mixture (local edit-KNN censored-location predictor
  with train-only abstention gate).
- Primary metric: junction-macro right-censored NLL (lower better).
- Status: NOT_PROMOTED (P4); DEVELOPMENT_ONLY. No mechanism claim.
- Prediction schema: per row mu, sigma, support, abstain (FinalPredictions.parquet).
