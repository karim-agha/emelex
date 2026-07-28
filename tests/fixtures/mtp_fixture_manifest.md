# MTP fixture manifests

Recorded 2026-07-25. Two records: the **dense BF16 fixture contract** backing v1
(dense BF16 only) and the **quantized/OptiQ evidence-gate record** seeding a
future quantized/MoE certification. Section 1 records the current
single-source, single-namespace contract; the superseded composed-fixture record
is preserved as Appendix A.

## 1. Dense BF16 fixture (v1) — converted, converter-native `language_model.mtp.*`

**Exactly one supported workflow: the converted fixture** (test-only in v1 — not a
supported end-user installation format; first-class distribution is an ADR follow-up).
Sidecar injection is a rejected alternative: there is no `mtp_file` config key,
no composition step, and no merged directory. The conversion runs on a
certification machine with the artifacts, automated by
`tools/mtp_fixture_convert.sh`; the accepted run is summarized below and any
deviation fails fixture acceptance.

### Pinned inputs

| Role | Source | Revision |
|---|---|---|
| **Weight source (single authoritative)** | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Converter | `AirRunner/mlx-lm` fork, `mlx_lm.convert` | `45f53582d64287aa875c1606e479f7f66c0afb58` |
| **Equivalence cross-check ONLY (never a weight source)** | `mlx-community/Qwen3.5-4B-MTP-bf16` | `c05eea475606a952730182a0308d05c7cf7ccd77` |

Weight-source facts (verified against the pinned revision): `text_config.
mtp_num_hidden_layers = 1`; the safetensors index already contains the complete
canonical `mtp.*` set (15 keys, root-level) alongside the backbone.

Cross-check artifact facts: `model.safetensors`, **241,200,628 bytes**, sha256
`ea2e38acd5abffb27510bfe00064ffc2e0186c9bbbd536daa568a081612fb1ab`;
`model_type = qwen3_5_mtp`, bare-ROOT MTP tensor keys only, no backbone, no
embed/head — **not independently loadable** (the loader's standalone-sidecar guard
returns a clear error on this directory).

### On-disk namespace rule

The converted fixture's on-disk MTP namespace is **exactly the 15-key
`language_model.mtp.*` set — converter-native, no post-conversion rename** — and it
is the ONLY namespace that can enable v1 MTP. The pinned fork retains MTP weights
during conversion whenever the config constructs an MTP head, and its sanitize
prepends `language_model.` to every key not already under
`model.language_model.`/`language_model.`, so the converted output carries the MTP
tensors as `language_model.mtp.*` without any rename step. Canonical `mtp.*` exists
only in memory after loader canonicalization. Any bare-root key, any
`model.language_model.*` key, or any partial/mixed set **fails fixture acceptance**;
the loader's forbidden-namespace guard independently keeps such layouts disabled
(warn-and-skip, backbone unchanged).

### Workflow (executed by `tools/mtp_fixture_convert.sh`)

1. **Download inputs** at the pinned revisions (target repo full; cross-check
   artifact's `model.safetensors` only, byte-count + sha256 verified against the
   pins above before use).
2. **Convert**: pinned fork @ `45f53582d64287aa875c1606e479f7f66c0afb58`,
   `python -m mlx_lm convert` (BF16, no
   quantization). Record the exact command line, fork commit, `mlx` / `mlx-lm` /
   Python versions, and the sha256 of every output file.
3. **Post-conversion namespace inspection (before any use)**: record the actual
   converter output key set and assert the MTP key/shape/dtype set equals exactly
   the 15-key `language_model.mtp.*` set below — all BF16, no bare-root keys, no
   `model.language_model.*` keys, no partial/mixed sets, no injection artifacts.
4. **Equivalence cross-check (mandatory)**: map each standalone-artifact bare-root
   tensor `K` to converted `language_model.mtp.K`; record per-tensor byte-equality
   or max-abs-difference. **Unexplained mismatch aborts fixture acceptance** —
   divergent weight provenance would invalidate the parity gate.
5. **Compatibility checks**: hidden 2560; vocab 248,320; gated shapes — q_proj
   `[8192, 2560]` (2 × 16 heads × 256), o_proj `[2560, 4096]`;
   `mtp_use_dedicated_embeddings == false`; exactly one MTP layer. Any mismatch
   fails fixture acceptance.
6. **Strip-mtp variant**: `tools/mtp_fixture_strip.py` (see below).

### The 15-key on-disk MTP set

Exactly these keys, each dtype BF16, with shapes recorded by the accepted
conversion:

```
language_model.mtp.fc.weight                                   # [2560, 5120] (2H -> H)
language_model.mtp.pre_fc_norm_embedding.weight                # [2560]
language_model.mtp.pre_fc_norm_hidden.weight                   # [2560]
language_model.mtp.norm.weight                                 # [2560]
language_model.mtp.layers.0.input_layernorm.weight             # [2560]
language_model.mtp.layers.0.post_attention_layernorm.weight    # [2560]
language_model.mtp.layers.0.self_attn.q_proj.weight            # [8192, 2560] gated
language_model.mtp.layers.0.self_attn.k_proj.weight            # [1024, 2560]
language_model.mtp.layers.0.self_attn.v_proj.weight            # [1024, 2560]
language_model.mtp.layers.0.self_attn.o_proj.weight            # [2560, 4096]
language_model.mtp.layers.0.self_attn.q_norm.weight            # [256]
language_model.mtp.layers.0.self_attn.k_norm.weight            # [256]
language_model.mtp.layers.0.mlp.gate_proj.weight               # [9216, 2560]
language_model.mtp.layers.0.mlp.up_proj.weight                 # [9216, 2560]
language_model.mtp.layers.0.mlp.down_proj.weight               # [2560, 9216]
```

No embed/head keys — shared with the backbone. Norm convention: converted
orientation (`conv1d.weight` last dim == 1), plain BF16 norm vectors; per the
loader contract the loader NEVER adds 1 to any norm weight.

### Strip-mtp variant (non-MTP regression mechanism)

`tools/mtp_fixture_strip.py` deterministically removes **exactly the enumerated 15
`language_model.mtp.*` keys** from a copy of the converted output, rewriting the
safetensors shards and `model.safetensors.index.json`, and verifies it removed no
non-MTP key (input keys == output keys + exactly those 15). Output digests are
recorded. This variant must load with `supports_mtp() == false` and behave
byte-identically to a never-MTP model. Misuse (non-MTP key removed / partial strip)
is a verification failure. Both conversion and stripping reject duplicate JSON
keys, unsupported dtypes, invalid shapes or byte lengths, non-contiguous tensor
ranges, duplicate cross-shard owners, incomplete standalone mappings, and index
ownership or `total_size` drift. Machine-readable records use logical
`<workdir>`/source labels rather than private machine paths.

### `EMELEX_TEST_MODEL` semantics

Points at the converted directory. The loader sees on-disk `language_model.mtp.*`
on a converted backbone — matched by the sole supported prefix, canonicalized in
memory to `mtp.*`; the raw-HF predicate is false by construction (converter output
uses `language_model.*`, not `model.language_model.*`, with converted conv1d
orientation); the forbidden-namespace guard is not triggered (no bare-root or
`language_model.model.mtp.*` keys exist per the namespace inspection).

### Parity interaction

The Python dump script and the Rust validation consume the **identical converted
directory** (same digests, same on-disk tensor set — parity compares the same
weights by construction). The fork's load path for a converted directory and the
first-step/recursive dump points must be re-confirmed against this 4B converted
output during certification and recorded in the parity manifest (Appendix A's feasibility
notes were read from the 9B module repo and are indicative, not binding).

### Negative expectations (tested)

- Standalone artifact directory (`model_type = qwen3_5_mtp`) → clear `Model::load`
  error naming the condition, never a partial load.
- Raw-HF directory → raw-HF predicate skips MTP; backbone unchanged.
- Wrong-namespace output (bare-root, `model.language_model.*`,
  `language_model.model.mtp.*`, or mixed) → fixture acceptance failure AND loader
  warn-and-skip.
- Strip-script misuse → verification failure.
- Dimension-incompatible artifacts → acceptance failure.

## 2. Quantized/OptiQ evidence-gate record (DEFERRED — not loadable in v1)

Read from `mlx-community/Qwen3.5-9B-OptiQ-4bit` @
`890b4c43f99ff392819d83605f7b1e59fa9688aa`, file `optiq/mtp.safetensors`
(185,118,739 bytes). v1's dense-BF16 guards keep this checkpoint at warn-and-skip
(scales/biases companions present). Recorded so the future evidence gate starts from
facts for a future explicit evidence gate:

- Discovery: main index has **zero** mtp keys; `config.json` points at the sidecar via
  `"mtp_file": "optiq/mtp.safetensors"` (subdirectory — invisible to `load_all`'s
  top-level listing). Config: `mtp_num_hidden_layers: 1`,
  `mtp_policy: "optiq-int4-prequantized-gs64"`, `mtp_tensor_count: 29`,
  `mtplx_mtp_quantization: {bits: 4, group_size: 64, mode: affine, prequantized: true}`,
  `text_config.mtp_use_dedicated_embeddings: false`, `text_config.attn_output_gate: true`.
  The main `quantization` override map contains **no** mtp entries — per-tensor
  resolution would need scales-presence + `mtplx_mtp_quantization`. (Note: the
  `mtp_file` sidecar-loading path was removed from the loader; this record
  documents the artifact's own layout, not a supported load path.)
- 29 tensors, bare `mtp.` prefix, no safetensors `__metadata__`: fusion norms +
  `mtp.norm` + `input/post_attention_layernorm` + `q/k_norm` all BF16;
  **`mtp.fc.weight` dense BF16 [4096, 8192]**; every attention/MLP projection
  U32-packed 4-bit affine with BF16 `scales` AND `biases` companions
  (e.g. `q_proj`: weight [8192, 512], scales [8192, 64], biases [8192, 64];
  `down_proj`: weight [4096, 1536], scales/biases [4096, 192]).
- Future-gate requirements this record does NOT yet satisfy: fold-mapping evidence
  (incl. companion-tensor folding) and a passing logit-parity run for the quantized
  layout class.

## Appendix A. SUPERSEDED — composed 9B fixture (historical record only)

> Superseded by § 1. The composition workflow below encodes the **rejected**
> sidecar-injection alternative: the `mtp_file` config key no
> longer exists in the loader, bare-root/unprefixed MTP namespaces are forbidden
> on-disk layouts, and this procedure MUST NOT be executed. Pins and digests are
> preserved because they were read directly from the named revisions.

| Source | Revision (repo SHA) | Historical role |
|---|---|---|
| `mlx-community/Qwen3.5-9B-MLX-bf16` | `d3b9dc1f346d744d22c6a22fcfcf03702cbe0124` | backbone (760 BF16 tensors, 0 mtp keys) |
| `mlx-community/Qwen3.5-9B-MTP-bf16` | `22de6695c52eb55bda931f99c44b087c454009d3` | standalone MTP module |

MTP module file digest (`model.safetensors`, 486,582,779 bytes, sha256):
`c97f1cbac2bef846a2f689108f70ca88bf0d91c4482c46621a86a3ca55dea208`

The superseded procedure copied the module file into the backbone dir as
`mtp.safetensors` and added `"mtp_file": "mtp.safetensors"` +
`"mtp_num_hidden_layers": 1` to `config.json`. The module repo
(`model_type = qwen3_5_mtp`) ships the 15-tensor dense set (fusion norms + fc + one
gated full-attention layer with q/k norms + final norm; no embed/head) with
**unprefixed** keys and `block_size: 3` (publisher's recommended draft depth;
informational only). Norm-convention evidence: converted-orientation lineage
(`conv1d.weight` last dim == 1), plain BF16 norm vectors — never add 1.

Parity-script feasibility (read from the 9B module + pinned fork @
`45f53582d64287aa875c1606e479f7f66c0afb58`):
`model.language_model.mtp_forward(hidden_states, next_token_ids, mtp_cache)` returns
per-step logits `(B, N, vocab)` — first-step parity directly scriptable. The recycle
hidden is internal to `MTPModule.__call__`, but every sublayer is a public attribute
(`...mtp.{pre_fc_norm_embedding, pre_fc_norm_hidden, fc, layers, norm}`), so a dump
script can reproduce the recursion without patching the fork. The fork hard-errors
when config promises MTP but weights lack it (inverse of emelex's warn-and-skip;
both internally consistent). Re-confirm against the 4B converted directory
during the § 1 parity interaction.

## Amendments

- 2026-07-25: the backbone repo in Appendix A is pinned by
  its git revision SHA `d3b9dc1f...` alone, without per-file digests. Rationale: the
  revision SHA is a content-addressed commit over the repo's LFS pointers, so it pins
  every backbone file transitively.
- 2026-07-25: § 1 rewritten to the current dense fixture contract —
  single authoritative weight source (`Qwen/Qwen3.5-4B@851bf6e8` via the pinned
  converter, converter-native `language_model.mtp.*`), standalone artifact demoted to
  mandatory equivalence cross-check, sidecar injection and post-conversion renaming
  recorded as rejected alternatives, composed-fixture record moved to Appendix A.

## Recorded acceptance evidence (2026-07-26)

- Acceptance: **PASS** (namespace inspection, equivalence cross-check,
  compatibility, strip variant all green).
- Versions: Python 3.10.10, mlx 0.32.0,
  mlx_lm 0.31.3 (fork @
  `45f53582d64287aa875c1606e479f7f66c0afb58`).
- Conversion: `python3 -m mlx_lm convert --hf-path <workdir>/target-851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --mlx-path <workdir>/converted --dtype bfloat16`
- Equivalence cross-check: **byte_equal: true for all 15 tensors**
  (converter output is byte-identical to the standalone artifact).
- config.json sha256 `77025ed0b68732aea52fd24d5e537e3838838cac8fe4e920735266c6f7c4e454`
- `model-00001-of-00002.safetensors` sha256 `76eaf5f25a656064e175b5d2538cbf03ad1bf2917f4cb76dd08febf81b811801`
- `model-00002-of-00002.safetensors` sha256 `72b17f015dd762fe07db7668a772852c5ebd4652e0374433a0a836ae012fb83a`
- Full machine-readable record: `<workdir>/mtp_fixture_record.json`
  (workdir artifact; regenerate via `tools/mtp_fixture_convert.sh`).
- Checked-in machine-readable parity certification:
  `mtp_certification.json`.
- Parity gate: PASS — see `mtp_parity_manifest.md`; rerun with
  `EMELEX_TEST_MODEL=<workdir>/converted
  EMELEX_PARITY_GOLDENS=<workdir>/goldens tools/party.py`.
