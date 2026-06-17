# Pipeline Dashboard GUI

Launch the dashboard from the repository root using an environment that has
PySide6 installed, usually the same environment used by the manual ROI GUI:

```bash
conda run -n caiman python current/pipeline_dashboard_gui.py
```

The dashboard is a controller for `current/run_pipeline.py`. Heavy processing
still runs in a child process, so the dashboard window remains responsive while
Fiji, CaImAn, suite2p, or post-manual analysis steps run.

Typical use:

1. Set `Data root`.
2. Set `Conda` and `Fiji` paths if the defaults are not correct.
3. Choose a step group such as `00-05 premanual`, `manual GUI`, or
   `06-16 postmanual`.
4. Use `Plan` to print the resolved command without running steps.
5. Use `Start` to run the selected steps and monitor the live log.
6. Use `Stop` to terminate the child process.

The `Open manual GUI` button launches the existing manual ROI curation GUI via
the existing `manual` pipeline step. The manual GUI implementation and behavior
remain separate from this dashboard.

Default Fiji paths:

- macOS: `/Applications/Fiji.app`
- Linux workstation: `/home/yifei/Fiji/fiji-linux-x64`

The Fiji field stays editable. Use `Browse` to choose a different Fiji install,
then `Set default` to remember that path for future dashboard launches on the
same computer.

Priority order:

1. Fiji path saved with `Set default`
2. `FIJI_BIN` or `FIJI_PATH` environment variable
3. Platform default listed above
