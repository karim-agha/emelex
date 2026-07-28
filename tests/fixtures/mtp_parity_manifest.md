# MTP logit-parity manifest — dense BF16 class

Fixture and numerical rows recorded 2026-07-26 on the implementation machine.
Provenance metadata was freshly regenerated from the verified pinned checkout
on 2026-07-27; all three numerical row hashes reproduced exactly. **The
schema-v2 gate passed again on 2026-07-27 over exactly 3 steps (first + two
recursive).** The canonical, machine-readable certification is
`mtp_certification.json`; this document records its rationale and observed
deltas.

## Fixture

Converted per `tools/mtp_fixture_convert.sh` from the single
authoritative weight source `Qwen/Qwen3.5-4B@851bf6e8` via the pinned
fork; acceptance PASS with the mandatory equivalence cross-check
against `mlx-community/Qwen3.5-4B-MTP-bf16@c05eea47` **byte-equal on
every one of the 15 `language_model.mtp.*` tensors**. The conversion
workdir's `mtp_fixture_record.json` recorded config.json sha256
`77025ed0b68732aea52fd24d5e537e3838838cac8fe4e920735266c6f7c4e454`;
shards `76eaf5f25a656064…` / `72b17f015dd762fe…`. Strip-mtp variant
built and verified (removed exactly the 15 keys).

## Pins

- mlx-lm fork: `AirRunner/mlx-lm @ 45f53582d64287aa875c1606e479f7f66c0afb58`
  (package version 0.31.3). The generator verified a clean local checkout,
  exact full HEAD and origin, plus byte equality between all installed and
  source `mlx_lm` Python files outside source-only `examples/`. Canonical
  verified tree digest:
  `b8130c5a098083b1c5ce44bb1e120f5f6d28aab9886da30d134f91aeee931761`.
- mlx (Python): **0.32.0** (acknowledged via `MLX_PIN_ACK`)
- Python 3.10.10
- Dump: `tools/mtp_parity_dump.py --steps 3`, prompt "The quick brown
  fox jumps over the lazy dog. Explain why.", recycle hiddens captured
  from inside `mtp_forward` via the documented norm tap.

## Goldens (sha256)

These hashes, the config and both complete shard hashes, source revision,
converter fork, and MLX version are bound together under implementation id
`emelex-qwen3.5-mtp-dense-bf16-v1` in `mtp_certification.json`.

- `step0.npy` `1c7a420332767b69ae508ee3e728f4af61baea487a2b5e29b857236d01bcff7d`
- `step1.npy` `073d5f73e233b22677f2d079badc6304906e7b3c33455657afbeb217766d9ada`
- `step2.npy` `a13fa2611c59fba2b35b68cdd8eb09c35378aa1414d71f25a4af26abee0fe249`
- `meta.json` `da4ce37a2a5ef6895fdf107526f85d35a973d4de080c42d9dc368ff88b998b37`

## Observed deltas and the threshold revision

| step | max abs diff | mean abs diff | max abs logit | top-5 agreement |
|---|---|---|---|---|
| 0 | 0.2017 | 0.0359 | 18.25 | identical sets, identical order |
| 1 | 0.2500 | 0.0403 | 16.12 | identical sets (ranks 2/3 swapped) |
| 2 | 0.2656 | 0.0457 | 21.00 | identical sets, identical order |

The original provisional `max-abs <= 2e-2` sat BELOW one bf16 ulp at
the observed logit scale (ulp(16..32) = 0.125) and was therefore
unsatisfiable by any independent bf16 implementation. Revised gate,
recorded with its numerical rationale: **max-abs <= 0.5 (~4 ulps at
observed scale), mean-abs <= 0.1, and the unchanged semantic gate of
mutual top-5-in-top-8 containment** — which the data passes with the
top sets in fact identical. Observed deltas are 1–2 ulps with sub-ulp
means: pure hardware numerics between two independent kernel stacks,
with perfect semantic agreement.

## Reproduction

```sh
WORKDIR=/path/to/mtp-fixture-work
tools/mtp_fixture_convert.sh "$WORKDIR"
VENV="$WORKDIR/venv"
MLX_PIN_ACK=$("$VENV/bin/python" -c 'import mlx.core; print(mlx.core.__version__)') \
  "$VENV/bin/python" tools/mtp_parity_dump.py \
    --model "$WORKDIR/converted" --out "$WORKDIR/goldens-new" --steps 3
EMELEX_TEST_MODEL="$WORKDIR/converted" \
  EMELEX_PARITY_GOLDENS="$WORKDIR/goldens-new" \
  tools/party.py
```

`tools/party.py` hashes the certified config, both shards, metadata, and all
three golden rows before inference. It runs only the exact ignored parity
test in a fresh process group and kills that group after a hard 1,200-second
deadline. Missing inputs, ignored/skipped execution, hash drift, missing MTP,
timeout, or numeric parity failure returns nonzero. The generator refuses an
existing output directory and publishes a complete fresh golden set by one
same-filesystem rename, so a failed run cannot leave a mixed certification.

The 2026-07-27 regeneration command was the command above with
`EMELEX_PARITY_GOLDENS=$WORKDIR/goldens-new`. It completed in 8.9 seconds and
reproduced metadata hash `da4ce37a…` plus the three row hashes listed above.
The final party completed successfully in 70 seconds wall time including
release compilation; the exact three-step inference test took 10.46 seconds.
