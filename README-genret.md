# genret: the researcher's code (Floor 3)

Owner: researcher. Everything outside `referee/` (the evaluation steward's). Designs and predictions are in the lab
repository: `lab/predictions/genret-GR1.md` and `genret-GR2.md` (shakabuku-labs PR #10).

- `genret/data.py`: ESCI through the referee's loader only (this repo's `referee.data` once it lands, else
  s3-search's `referee/loader.py`, the one the steward pins).
- `genret/text.py`, `genret/embed.py`: product text and public sentence-encoder embeddings.
- `genret/rqvae.py`: TIGER's RQ-VAE (GR1). `genret/rqkmeans.py`, `genret/kmeans.py`: residual k-means (GR2, and GR1
  arm A5).
- `genret/sids.py`: TIGER's 4th code, collision statistics (the referee's definitions), the sid map, the prefix trie.
- `genret/beam.py`: trie-constrained beam search over fixed-depth ids.
- `scripts/tokenize_catalogue.py`: sid map + statistics for a dataset. `scripts/build_colab_tokenise_job.py`: the
  self-contained Colab job for the whole ESCI-US catalogue.

Tests: `python -m pytest -q tests` (CPU, seconds). Every number from esci-sample or a development encoder is
development only, never a claim.
