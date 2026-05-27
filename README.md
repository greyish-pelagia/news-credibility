# News Credibility MVP with LIAR

This repository is a compact machine learning MVP for binary credibility classification of short political/news-related claims from the LIAR dataset.

The project predicts credibility for short textual claims, not full articles. It does not perform real-time fact checking.

## Main workflow

1. Run `data_exploration.ipynb` to download, process, and explore LIAR.
2. Run `analysis.ipynb` to train and compare the models.
3. Review outputs in `reports/` and saved models in `models/`.

## Main artifacts

- `data/processed/liar_binary.csv`
- `reports/metrics_summary.csv`
- `reports/bertopic_topic_risk.csv`
- `reports/error_analysis.md`
