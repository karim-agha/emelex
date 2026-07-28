# Home context

Everything Emelex creates belongs beneath one canonical `EmelexHome`:

```text
config.toml
.emelex-root
models/
cache/
memory/emelex.sqlite3
temp/
.snapshot-mutations.lock
```

Home and database paths are authority boundaries, not configuration fields.
Project configuration cannot redirect them.

`.emelex-root` records that a nonempty directory is dedicated Emelex storage.
Existing broad/shared directories are rejected without permission changes.
Preparation holds parent and root directory descriptors, refuses symlinks,
checks current-user ownership plus `0700` and the absence of macOS extended
ACLs, and verifies canonical dev/inode identity before creating descendants.
An exclusive `flock` on the opened root directory serializes preparation
across processes and remains held through ownership-marker publication,
standard subdirectory preparation, and the root sync.
All ownership checks share one documented safe wrapper around `geteuid`.
New objects have inherited ACLs cleared through their open descriptors;
existing objects with ACL entries are never normalized silently.

Exact model binding and model-store mutation share an owner-only advisory lock.
Memory holds it across installed-snapshot validation and database commit;
model management holds it across reference checks and quarantine or deletion.
This closes cross-process check-versus-mutation races without opening SQLite
when memory has never been used.
