# Real-case acceptance

A candidate must have finite q loss strictly below `1e-3`, at least 90% of retained observed peaks matched, no NaN conditional uncertainties, and each conditional uncertainty at most 0.1 internally scaled length units or 1 degree. The optimizer status does not determine this quality flag.

For the serial hybrid schedule, the reported loss is the last simulated-fractional fixed-assignment optimization loss. Final coverage is recomputed by exact one-to-one assignment followed by the shared distance cutoff parameter 0.06 for the four real-case profiles (physical cutoff 0.054 Å⁻¹ at q_scale 0.90). Its denominator is the number of retained observed peaks, not the number of simulated reflections. Missing or extinguished simulated reflections do not count against coverage.

Checks use internally scaled coordinates, before physical-unit conversion. The synthetic recovery criterion against known cells remains unchanged (all lengths below 0.1 angstrom error and all angles below 0.1 degree error). Historical v1.0.0 validation used an inclusive `5e-5` loss threshold; its saved results are historical and are not overwritten. Changing acceptance does not change the fitted cells, but can change candidate ranking.
