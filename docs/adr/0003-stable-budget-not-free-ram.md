# Fit uses stable budget, not free RAM

Catalog and capacity fit badges use a stable planner budget: total unified memory minus configurable OS reserve, capped by Metal recommended working set when present. Current free RAM and browser pressure are shown separately and must not change fit badges, so the catalog does not flicker when other apps allocate memory.
