# Toolkit facade

`Emelex` resolves one invocation: canonical working directory, one Emelex
Home, and strict global/project configuration.

Static Hub discovery, Metal fit budget, owned-snapshot and external-link model
management, and durable memory are fallible lazy facets initialized at most
once per facade.
The static Hub facet makes no machine-fit claim and does not query Metal; the
model manager owns a separately profiled Hub client using the invocation's
actual workload and Metal budget. Its Hub searches probe current Emelex Home
filesystem availability; unrelated model-manager operations do not. Both
clients use the same invocation credential. Explicit builder
`HubCredentials` take precedence over an optional global `[hub].token`;
without either they are anonymous. The stored secret is extracted separately
and never appears in resolved `Config`. Environment lookup remains the
embedding application's responsibility.
Construction performs no network request, SQLite open, storage probe, Metal
query, runtime extraction, or MLX initialization. Network access starts only
through explicit Hub calls; MLX starts only through model load/verification.
