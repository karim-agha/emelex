# Artifact reads

This internal module is the sole low-level reader for model-owned configuration,
tokenizer, template, and checkpoint metadata. It opens final components with
`O_NOFOLLOW`, verifies a regular file through the opened descriptor, and reads
through `take(limit + 1)` so metadata races cannot bypass byte caps.

Callers choose a semantic limit and map I/O errors into their public domain
error. Native checkpoint loading separately keeps an opened descriptor alive
while MLX materializes tensors.
