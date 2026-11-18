# Interactive dashboard (Streamlit)

Four tabs: Overview (cleaning Sankey, sky chart), Models (radar, dumbbell, per-seed box plots, ablation violins,
results table), Star explorer (light curve coloured by A_k, per-window prediction heatmap) and Observers
(A_k violins, observer bubble chart, searchable table).

It reads only the tidy tables in `dashboard_data/`, which the pipeline's `dashboard` step writes to
`v3_5GB/dashboard_data/` (CSV copies are kept here so the app runs anywhere). Nothing is retrained.

```bash
pip install -r requirements.txt
streamlit run app.py
```

To use a newer run's tables, copy the CSVs from `v3_5GB/dashboard_data/` into `dashboard_data/`, or set
`COSMIC_DASHBOARD_DATA` to that folder.
