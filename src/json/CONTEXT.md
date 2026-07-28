# JSON module context

- `structurally_bounded` is iterative so hostile `serde_json::Value` input is
  rejected before serde's recursive serializer can consume it.
- Depth counts the root at zero. Exactly 64 nested containers are accepted;
  65 are rejected.
- Node count includes the root. Exactly 65,536 nodes are accepted.
- Callers still enforce their own encoded-byte limits.
