# JSON bounds

Shared iterative validation for caller-controlled JSON trees.

Emelex rejects values deeper than 64 containers or larger than 65,536 total
nodes before recursive serde serialization. Byte ceilings remain the
responsibility of each owning API because tool schemas, event payloads, and
snapshots have different limits.
