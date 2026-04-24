# Simulated GPSI Dataset

`simulated_gpsi_data.csv` is a small synthetic, de-identified dataset for quick
examples and tests.

Columns:

- `subject_id`: synthetic subject identifier.
- `true_cluster`: simulated subgroup label used only for evaluation.
- `true_stage`: simulated latent disease stage used only for evaluation.
- `feature_1` ... `feature_6`: noisy non-linear feature trajectories with a
  small amount of missingness.

The GPSI model is unsupervised. `true_cluster` and `true_stage` are not used by
the model in `examples/run_simulated.py`; they are included so users can inspect
whether inferred subgroups and stages recover the simulated structure.

Regenerate this file with:

```bash
python examples/generate_simulated_data.py \
  --out data/simulated/simulated_gpsi_data.csv \
  --n-subjects 120 \
  --n-features 6 \
  --n-subgroups 2
```

