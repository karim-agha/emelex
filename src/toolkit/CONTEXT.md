# Toolkit facade — context

## Invariants

- One immutable configuration snapshot per invocation.
- Invocation root and Emelex Home canonicalize before dependent services.
- Construction has no hidden network activity.
- Construction does not open SQLite or create/query a Metal device.
- Runtime initialization remains lazy and first-home-wins.
- Hub, model manager, memory, and Metal budget initialize independently on
  first access.
- Hub credentials are explicit per facade. The library reads no authentication
  environment variables; both lazy Hub clients receive the same redacted
  credential value when configured.
- The direct Hub facet is static-only. The model manager owns a fit-profiled
  Hub client with the same configuration bounds plus the facade's workload and
  Metal budget. That client probes Emelex Home filesystem availability only
  when searching; unrelated model-manager operations do not depend on the
  probe, and downloads recheck it.
- The model facet installs a lightweight durable-Session reference guard.
  Construction captures only the database path; SQLite remains unopened until
  quarantine or deletion needs a reference check.
