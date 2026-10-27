# Compact trained deployment artifacts

These small NumPy actor weights are intentionally versioned for reproducible gateway
deployment. They are exported from actual Stable-Baselines3 PPO checkpoints; they
are not handwritten heuristics. Each deployment metadata file includes the source
checkpoint hash and exported-array hash. Runtime loading verifies the exported
array hash, feature schema, normalization, and matrix shapes.

The original trainable PPO ZIP checkpoints, optimizer state, TensorBoard logs, and
periodic checkpoints remain in ignored `models/` directories. Reproduce them with
the commands in `docs/rl.md`. `policy.npz` uses numeric arrays and is loaded with
`allow_pickle=False`. PyTorch is needed for training, not for deploying these actors.

Resumable selected-model checkpoints are also downloadable from the
[v0.2.1 release](https://github.com/xMN-28/Reinforcement-Learning-Based-API-Rate-Control/releases/tag/v0.2.1).

The optional sklearn predictor is a trusted repository-generated joblib artifact;
load only artifacts you trust. Its sidecar records feature semantics and checksum.
