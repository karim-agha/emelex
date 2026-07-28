# Artifact context

- Never use `std::fs::read`, `read_to_string`, or path-based third-party
  constructors for model-owned files.
- Read bytes through one `O_NOFOLLOW` descriptor.
- Enforce limits from actual bytes read, not only metadata.
- Parse tokenizer JSON from bounded bytes with `Tokenizer::from_bytes`.
- This module protects resource and path integrity; installed-model manifest
  verification supplies cryptographic integrity.
