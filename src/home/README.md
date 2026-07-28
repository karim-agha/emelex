# Home

`EmelexHome` resolves the only root Emelex may own. Precedence:

1. explicit library path or CLI `--home`;
2. `EMELEX_HOME`;
3. `~/.emelex`.

The selected path must be a dedicated leaf. A new leaf is created mode `0700`;
an existing leaf must already be owner-owned mode `0700` and must either be
empty or contain Emelex's valid `.emelex-root` marker. Emelex never chmods or
adopts a shared nonempty directory. Filesystem roots are always rejected.

The root and standard descendants are opened and created descriptor-relative
with `O_NOFOLLOW`. New objects have inherited extended ACLs removed; existing
objects with any extended ACL are rejected without mutation. Device/inode
identity is checked before publication. Preparation holds an exclusive
advisory lock on the opened root directory through marker creation, standard
subdirectory creation, and the final sync, so another cooperative process
never validates the marker or layout while publication is in progress. No XDG
or `Application Support` fallback exists.

The hidden owner-only `.snapshot-mutations.lock` serializes durable Session
binding with model quarantine and deletion across processes.
