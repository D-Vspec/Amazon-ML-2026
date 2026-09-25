# Docs

- [normalization.md](normalization.md): how names and addresses are transliterated and normalized, and how that was validated against the ground truth.
- [data_splits.md](data_splits.md): the training data split into `N_SETS` (default 10) equal, cluster-aware sets, and how to run on them.
- [evaluation.md](evaluation.md): the leaderboard's macro F0.5 and blocking recall, computed locally, and how much precision matters.
- [blocking.md](blocking.md): the TF-IDF and multilingual-embedding blockers and their union, measured recall@k, GPU vs CPU speed, and what each one misses.
- [matching.md](matching.md): pair features, the XGBoost matcher and threshold, and F0.5 of `model.json` vs the retrained union models (`model_union.json`).
