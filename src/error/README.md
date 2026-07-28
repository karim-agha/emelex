# Error

The non-exhaustive public `Error` classifies:

- home/runtime preparation (`Home`, `Runtime`);
- checkpoint selection and loading (`ModelPath`, `ModelLoad`);
- caller-correctable requests (`UnsupportedContent`, `InvalidRequest`,
  `InvalidConfiguration`, `InvalidParams`);
- engine execution and terminal stream reconciliation (`Generation`,
  `StreamProtocol`, `InferencePanic`);
- bounded worker transport (`InferenceBusy`, `InferenceChannel`).

Native APIs return this enum directly. With `rig`, request/configuration
variants map to `CompletionError::RequestError`; runtime, model, generation,
stream-protocol, panic, busy, and transport variants map to `ProviderError`.
