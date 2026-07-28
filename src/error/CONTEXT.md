# error — context

## Invariants

- Provider code never panics: the workspace denies
  `unwrap`/`expect`/`panic` outside the vendored engine subtree, and
  every fallible path funnels into `Error` or rig's `CompletionError`.
- Error classification rule: anything the *caller* could fix by changing
  the request maps to `CompletionError::RequestError`
  (`UnsupportedContent`, `InvalidRequest`, `InvalidConfiguration`,
  `InvalidParams`); anything environmental, saturated, or engine-side maps to
  `CompletionError::ProviderError`.
- `InferenceBusy` means bounded queue admission failed without waiting.
  `InferenceChannel` means the worker disconnected during submit/receive.
- `StreamProtocol` means incremental answer text was not an exact prefix of
  the terminal answer and therefore could not be reconciled losslessly.
- `Error` is `#[non_exhaustive]`: downstream matches must keep a
  wildcard arm, allowing new failure modes without a breaking change.
