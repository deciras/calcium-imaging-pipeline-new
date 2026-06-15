# Current Code Layout

`run_pipeline.py` is the main launcher. Step scripts are grouped by role:

- `preprocess/`: data organization, Fiji conversion, stimulus maps, motion correction, and spatial high-pass movies.
- `roi/`: suite2p ROI candidate generation and the manual ROI curation GUI.
- `analysis/`: post-manual trace extraction, response analysis, population analysis, summaries, and reports.
- `tools/`: maintenance utilities that are not part of the normal numbered pipeline.

Use `python3 current/run_pipeline.py --list-steps` to see the runnable steps.

Useful step groups:

- `premanual`: steps 00-05, through suite2p ROI candidate generation.
- `manual`: open the manual ROI curation GUI.
- `basic-analysis`: steps 06-09, from final ROI trace extraction through angle tuning.
- `postmanual`: steps 06-16, all automated analysis after manual ROI curation.

For GUI integration, `basic-analysis` can be limited to one trial with `--trial-id`.

Manual GUI notes:

- Step `manual` opens `roi/05_manual_roi_curation_gui.py`.
- The GUI supports synchronized zoom/pan across the reference and edit views:
  - `Ctrl` / `Cmd` + wheel zooms.
  - Pinch gestures zoom on supported trackpads.
  - Wheel / two-finger scroll pans after zooming.
  - `drag to pan` enables left-button panning; middle-button panning also works.
- The view can show both images, only the reference view, or only the edit view, and the two image positions can be swapped.
- Frame rendering caches the normalized base frame and static ROI overlays, so playback and frame scrubbing do not redraw every suite2p ROI on every frame.
- Manual ROI traces are updated lazily after final ROI set edits and forced current before saving.
- Undoing a just-finished manual ROI clears the finished drawing preview line as well as removing the ROI.
- The GUI has explicit `Save and close` and `Close` buttons. Closing the window also asks whether to save unsaved ROI edits.
