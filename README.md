# Olympus Calcium Imaging Pipeline

This repository contains code for Euprymna / Sepia retina calcium imaging data processing.

Real imaging data are stored on the Linux workstation and are not included in this repository.

Current maintained code is in:

- current/

Historical dated folders are kept locally but should not be used as the main pipeline unless explicitly needed.

Main goal:
- make the suite2p-based calcium imaging pipeline configurable, skippable, and safe for large datasets
- avoid rerunning completed steps
- avoid overwriting or deleting real data accidentally

Important:
- Do not commit raw or processed imaging data.
- Do not commit .oir, .tif, .tiff, .npy, suite2p output folders, or large movie files.
- Default running mode should be skip, not overwrite.
