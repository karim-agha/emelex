# Toolkit facade

`Emelex` resolves one invocation: canonical working directory, one Emelex
Home, and strict global/project configuration.

Static Hub discovery, Metal fit budget, immutable model management, and
durable memory are fallible lazy facets initialized at most once per facade.
The static Hub facet makes no machine-fit claim and does not query Metal; the
model manager owns a separately profiled Hub client using the invocation's
actual workload and budget. Both are anonymous unless the builder receives
explicit `HubCredentials`; environment lookup remains the embedding
application's responsibility. Construction performs no network request, SQLite
open, Metal query, runtime extraction, or MLX initialization. Network access
starts only through explicit Hub calls; MLX starts only through model
load/verification.
