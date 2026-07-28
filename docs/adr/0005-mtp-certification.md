# ADR 0005: Byte-bound MTP certification

## Status

Accepted on 2026-07-26.

## Context

Some Qwen checkpoints publish multi-token-prediction weights that can act as a
self-draft model for speculative decoding. Repository tags and plausible
tensor names do not prove that a loader interpretation produces distribution-
equivalent output. Silently falling back from requested speculation would also
make capability reporting and performance measurements misleading.

A release gate must catch numerical or loader drift without becoming an
unbounded multi-hour test.

## Decision

Production MTP enablement is limited to a precisely described, reviewed layout.
The runtime creates one immutable checkpoint snapshot: a pinned model-directory
descriptor, owned `config.json` bytes, and already-opened descriptors for the
exact selected shards. Config and shards are opened relative to that directory
descriptor. Model construction, MLX materialization, and certification all
consume that same snapshot; none reopens a model-owned pathname. This closes
rename, A→B→A, mid-snapshot root replacement, and whole-directory swap races
between load and certification. The runtime then:

1. validates the architecture and every required tensor name, dtype, and
   shape;
2. hashes the snapshot's exact configuration and private, unlinked weight
   shards once before load, then revalidates each still-open descriptor's
   identity and header/layout after materialization;
3. matches those bytes against the checked-in certification manifest; and
4. advertises runtime-verified MTP only after the model loads successfully.

Nonzero speculative-token configuration fails with a typed capability error
when the exact checkpoint is not certified, and initial MTP priming failure
fails the request rather than pretending speculation started. Once target
decoding has a valid committed prefix, an isolated MTP forward or commit
failure discards only the MTP state and continues target-only. Per-call
speculation accounting reports work that actually occurred; recovered MTP
failures never corrupt or reuse the discarded MTP state.

`tools/party.py` is the external certification gate. It validates required
inputs and hashes, executes exactly three recorded parity steps, requires a
completion sentinel, and applies a hard 1,200-second process-group deadline.
Missing fixtures, a skipped or filtered test, hash drift, timeout, absent MTP,
or numerical mismatch is failure.

Ordinary unit tests do not require the large external fixture. They validate
the manifest, strict layout class, loader boundary, and failure behavior with
small fixtures. `tools/test_mtp_fixture_safetensors.py` is a required check for
the evidence tooling itself: duplicate JSON keys or tensor owners, invalid
dtype/shape/byte-length relationships, ambiguous payload intervals, incomplete
standalone mappings, and stale shard indexes fail closed.

## Consequences

- `acceleration:mtp` means more than a model-card claim.
- The supported MTP class is intentionally narrower than all plausible Qwen
  variants.
- A new dtype, quantization, architecture, or layout needs its own evidence and
  certification before production enablement.
- The parity party has a deterministic maximum wall-clock budget of 20 minutes.

Golden generation is fail-closed too. The generator accepts only the pinned
mlx-lm package installed from a clean local git checkout with the expected
origin and full commit SHA, verifies installed Python bytes against that
checkout, records the verified package/tree identity without machine-local
paths, and atomically publishes into a previously absent output directory.
- Capability selection and initial priming never silently downgrade a requested
  speculative call; isolated mid-decode MTP faults use the explicit
  discard-and-continue transition above.
