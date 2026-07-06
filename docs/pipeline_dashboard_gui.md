# Pipeline Dashboard GUI

Launch the dashboard from the repository root using an environment that has
PySide6 installed, usually the same environment used by the manual ROI GUI:

```bash
conda run -n caiman python current/pipeline_dashboard_gui.py
```

On the Linux workstation, use the dedicated lightweight dashboard environment:

```bash
bash linux_workstation/launch_pipeline_dashboard.sh "$DATA_ROOT"
```

That launcher defaults to `DASHBOARD_ENV=dashboard_gui` and
auto-detects a common `conda` path unless you override `CONDA_BIN`.

The dashboard is a controller for `current/run_pipeline.py`. Heavy processing
still runs in a child process, so the dashboard window remains responsive while
Fiji, CaImAn, suite2p, or post-manual analysis steps run.

Typical use:

1. Set `Data root`.
2. Set `Conda` and `Fiji` paths if the defaults are not correct.
3. Choose a step group such as `00-05 premanual`, `manual GUI`, or
   `06-18 postmanual`.
4. Use `Plan` to print the resolved command without running steps.
5. Use `Start` to run the selected steps and monitor the live log.
6. Use `Stop` to terminate the child process.

For post-manual population analysis, `Analysis input` controls the matrix used
by step 13 similarity, step 14 hierarchical clustering, and step 16 PCA/UMAP:

- `Stimulus slices` is the recommended default. It uses step 12 normalized
  peri-stimulus slice features, so clustering is based on stimulus-locked
  response signatures rather than the entire continuous trace.
- `Full traces` uses complete dF/F traces and is useful as a QC view for drift,
  bleaching, spontaneous waves, or tissue-state effects.
- `Summary features` uses scalar ROI metrics from step 11.
- `Response scalars only` uses compact response matrices where supported.

`Cluster scaling` controls step 14 only: normalized, raw/source-scale, or both.

The dashboard remembers the last `Data root` used by `Plan`, `Start`, or
`Open manual GUI`, so the next launch opens at the previous working location.

The `Open manual GUI` button launches the existing manual ROI curation GUI via
the existing `manual` pipeline step. The manual GUI implementation and behavior
remain separate from this dashboard.

Default Fiji paths:

- macOS: `/Applications/Fiji.app`
- Linux: `~/Fiji`, `~/Fiji.app`, `/opt/Fiji`, `/opt/Fiji.app`, `/usr/local/Fiji`

Default Fiji memory:

- macOS: `16G`
- Linux workstation: `64G`

The Fiji field stays editable. Use `Browse` to choose a different Fiji install,
then `Set default` to remember that path for future dashboard launches on the
same computer.
On macOS, saving `/Applications/Fiji.app` is OK; the conversion step resolves it
to the executable inside the app bundle before launching Fiji.

Priority order:

1. Fiji path saved with `Set default`
2. `FIJI_BIN` or `FIJI_PATH` environment variable
3. Platform default listed above
