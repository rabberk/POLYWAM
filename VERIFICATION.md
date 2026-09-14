# Verification scope

- Export uses an explicit source-file allowlist; no original Git history.
- Python source syntax parsed with `ast.parse`.
- Training shell launcher checked with `bash -n`.
- CPU smoke tests passed: action-block shape and gradient scale, causal prefix
  visibility, dense Transformer output shape and finite/nonzero input gradients.
- Core tests use PyTorch 2.x; they do not load V-JEPA/Qwen weights or data.
- End-to-end CUDA/FSDP training has not been rerun for this anonymous snapshot.
- This artifact has not yet been published to an anonymous hosting service.
