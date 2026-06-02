#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desktop GUI for manual ROI curation after suite2p.

The GUI is intentionally conservative:
- it does not edit suite2p outputs in place
- rejecting an existing ROI only writes a new manual iscell file
- drawing a ROI writes separate manual ROI files
"""

from __future__ import annotations

import argparse
import copy
import json
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile as tf
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from PySide6.QtCore import QPoint, QPointF, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


STEP_NAME = "05e_roi_manual_curation"


@dataclass
class TrialPaths:
    trial_id: str
    suite2p_dir: Path | None
    plane0_dir: Path | None
    stat_path: Path | None
    iscell_path: Path | None
    f_path: Path | None
    fneu_path: Path | None
    curated_iscell_path: Path | None
    movie_path: Path
    output_dir: Path


def find_trial_paths(data_root: Path, trial_id: str | None, movie_kind: str) -> TrialPaths:
    suite2p_root = data_root / "05_suite2p_roi_detection"
    stat_paths = sorted(suite2p_root.rglob("suite2p/plane0/stat.npy"))
    if stat_paths and trial_id is None:
        stat_path = stat_paths[0]
    elif stat_paths and trial_id is not None:
        matches = [path for path in stat_paths if path.parent.parent.parent.name == trial_id]
        stat_path = matches[0] if matches else None
    else:
        stat_path = None

    if stat_path is not None:
        plane0_dir = stat_path.parent
        suite2p_dir = plane0_dir.parent.parent
        trial_id = suite2p_dir.name
        iscell_path = plane0_dir / "iscell.npy"
        f_path = plane0_dir / "F.npy"
        fneu_path = plane0_dir / "Fneu.npy"
    else:
        if trial_id is None:
            raise FileNotFoundError(
                "No suite2p stat.npy found. Please pass --trial-id so the GUI can open a movie-only trial."
            )
        plane0_dir = None
        suite2p_dir = None
        iscell_path = None
        f_path = None
        fneu_path = None

    curated = data_root / "05c_roi_quality_filter" / trial_id / f"{trial_id}_iscell_curated.npy"
    movie_path = find_movie_path(data_root, trial_id, movie_kind)
    return TrialPaths(
        trial_id=trial_id,
        suite2p_dir=suite2p_dir,
        plane0_dir=plane0_dir,
        stat_path=stat_path,
        iscell_path=iscell_path,
        f_path=f_path,
        fneu_path=fneu_path,
        curated_iscell_path=curated if curated.exists() else None,
        movie_path=movie_path,
        output_dir=data_root / STEP_NAME / trial_id,
    )


def find_movie_path(data_root: Path, trial_id: str, movie_kind: str) -> Path:
    if movie_kind == "raw":
        root = data_root / "01_oir_to_tif" / trial_id
        matches = sorted(root.glob("*_Max_Proj.tif"))
        if matches:
            return matches[0]
    if movie_kind == "spatial-highpass":
        root = data_root / "04_spatial_highpass" / trial_id
        matches = sorted(root.glob("*_spatial_highpass_movie.tif"))
        if matches:
            return matches[0]
    if movie_kind in {"corrected", "spatial-highpass"}:
        root = data_root / "03_motion_correct" / trial_id
        matches = sorted(root.glob("*_corrected_movie.tif"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No movie found for {trial_id} using movie_kind={movie_kind}")


def read_index_csv(path: Path) -> set[int]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    values: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("suite2p_original_id"):
                continue
            try:
                values.add(int(float(text.split(",")[0])))
            except ValueError:
                continue
    return values


def discover_trial_ids(data_root: Path, movie_kind: str) -> list[str]:
    roots: list[Path] = []
    if movie_kind == "raw":
        roots.append(data_root / "01_oir_to_tif")
    if movie_kind == "spatial-highpass":
        roots.append(data_root / "04_spatial_highpass")
    if movie_kind in {"corrected", "spatial-highpass"}:
        roots.append(data_root / "03_motion_correct")
    trial_ids: list[str] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        if root.name == "01_oir_to_tif":
            pattern = "*_Max_Proj.tif"
        elif root.name == "04_spatial_highpass":
            pattern = "*_spatial_highpass_movie.tif"
        else:
            pattern = "*_corrected_movie.tif"
        for movie in sorted(root.glob(f"*/{pattern}")):
            trial_id = movie.parent.name
            if trial_id not in seen:
                seen.add(trial_id)
                trial_ids.append(trial_id)
    return trial_ids


def load_iscell(paths: TrialPaths, n_roi: int) -> np.ndarray:
    if paths.iscell_path is None:
        out = np.zeros((n_roi, 2), dtype=np.float32)
        out[:, 1] = np.nan
        return out
    if paths.curated_iscell_path is not None:
        arr = np.load(paths.curated_iscell_path, allow_pickle=True)
    else:
        arr = np.load(paths.iscell_path, allow_pickle=True)
    if arr.ndim != 2 or arr.shape[0] != n_roi:
        out = np.zeros((n_roi, 2), dtype=np.float32)
        out[:, 1] = np.nan
        return out
    if arr.shape[1] == 1:
        out = np.zeros((n_roi, 2), dtype=np.float32)
        out[:, 0] = arr[:, 0]
        out[:, 1] = np.nan
        return out
    return arr.astype(np.float32, copy=True)


def movie_as_tyx(path: Path) -> np.ndarray:
    arr = tf.memmap(path)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr.reshape((1, arr.shape[-2], arr.shape[-1]))
    return arr.reshape((-1, arr.shape[-2], arr.shape[-1]))


def normalize_frame(frame: np.ndarray, low: float, high: float) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32)
    finite = frame[np.isfinite(frame)]
    if finite.size == 0:
        return np.zeros(frame.shape, dtype=np.uint8)
    vmin, vmax = np.percentile(finite, [low, high])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
    scaled = (frame - vmin) / max(vmax - vmin, 1e-6)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def mask_bounds(ypix: np.ndarray, xpix: np.ndarray) -> tuple[float, float, float]:
    if len(ypix) == 0:
        return np.nan, np.nan, np.nan
    cx = float(np.mean(xpix))
    cy = float(np.mean(ypix))
    radius = float(max(np.std(xpix), np.std(ypix), 3.0) * 2.0)
    return cx, cy, radius


def robust_dff(fcorr: np.ndarray, percentile: float, eps: float) -> tuple[np.ndarray, float]:
    finite = fcorr[np.isfinite(fcorr)]
    if finite.size == 0:
        return np.zeros_like(fcorr, dtype=np.float32), float("nan")
    f0 = float(np.percentile(finite, percentile))
    denom = max(abs(f0), eps)
    return ((fcorr - f0) / denom).astype(np.float32), f0


def polygon_mask(points: list[tuple[float, float]], shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    mask = MplPath(points).contains_points(coords)
    return mask.reshape((height, width))


def ellipse_points(start: tuple[float, float], end: tuple[float, float], n_points: int = 72) -> list[tuple[float, float]]:
    x0, y0 = start
    x1, y1 = end
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    rx = abs(x1 - x0) / 2.0
    ry = abs(y1 - y0) / 2.0
    if rx < 1.0 or ry < 1.0:
        return []
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_points), endpoint=False)
    return [(float(cx + rx * np.cos(t)), float(cy + ry * np.sin(t))) for t in theta]


def annulus_mask(roi_mask: np.ndarray, inner_px: int = 3, outer_px: int = 12) -> np.ndarray:
    try:
        from scipy import ndimage as ndi
    except Exception:
        return np.zeros_like(roi_mask, dtype=bool)

    outer = ndi.binary_dilation(roi_mask, iterations=max(int(outer_px), 1))
    inner = ndi.binary_dilation(roi_mask, iterations=max(int(inner_px), 1))
    return outer & ~inner


def suite2p_roi_boundary(ypix: np.ndarray, xpix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return boundary pixels so suite2p ROIs keep their real shape on screen."""
    if ypix.size == 0 or xpix.size == 0:
        return ypix, xpix
    y0, y1 = int(np.min(ypix)), int(np.max(ypix))
    x0, x1 = int(np.min(xpix)), int(np.max(xpix))
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
    mask[ypix - y0, xpix - x0] = True
    try:
        from scipy import ndimage as ndi

        inner = ndi.binary_erosion(mask, iterations=1, border_value=0)
        boundary = mask & ~inner
    except Exception:
        boundary = mask
    by, bx = np.nonzero(boundary)
    return by + y0, bx + x0


def stat_dict_from_mask(mask: np.ndarray, roi_id: int) -> dict:
    ypix, xpix = np.nonzero(mask)
    lam = np.ones(len(xpix), dtype=np.float32)
    med = np.array([float(np.mean(ypix)), float(np.mean(xpix))], dtype=np.float32) if len(xpix) else np.array([np.nan, np.nan])
    return {
        "ypix": ypix.astype(np.int32),
        "xpix": xpix.astype(np.int32),
        "lam": lam,
        "med": med,
        "npix": int(len(xpix)),
        "overlap": np.zeros(len(xpix), dtype=bool),
        "manual_roi_id": int(roi_id),
    }


def imagej_polygon_roi_bytes(points: list[tuple[float, float]], roi_name: str = "") -> bytes:
    if len(points) < 3:
        raise ValueError("ImageJ ROI needs at least 3 points")
    xs = np.asarray([p[0] for p in points], dtype=float)
    ys = np.asarray([p[1] for p in points], dtype=float)
    left = int(np.floor(xs.min()))
    top = int(np.floor(ys.min()))
    right = int(np.ceil(xs.max())) + 1
    bottom = int(np.ceil(ys.max())) + 1
    rel_x = np.clip(np.rint(xs - left), 0, 65535).astype(">u2")
    rel_y = np.clip(np.rint(ys - top), 0, 65535).astype(">u2")
    n = int(len(points))
    header = bytearray(64)
    header[0:4] = b"Iout"
    struct.pack_into(">H", header, 4, 227)
    header[6] = 0  # polygon
    struct.pack_into(">hhhhH", header, 8, top, left, bottom, right, n)
    return bytes(header) + rel_x.tobytes() + rel_y.tobytes()


def safe_roi_name(prefix: str, roi_id: int) -> str:
    return f"{prefix}_{roi_id:04d}.roi"


def ordered_boundary_points(ypix: np.ndarray, xpix: np.ndarray) -> list[tuple[float, float]]:
    by, bx = suite2p_roi_boundary(np.asarray(ypix, dtype=np.int32), np.asarray(xpix, dtype=np.int32))
    if len(bx) < 3:
        return [(float(x), float(y)) for y, x in zip(by, bx)]
    cx = float(np.mean(bx))
    cy = float(np.mean(by))
    order = np.argsort(np.arctan2(by - cy, bx - cx))
    return [(float(bx[i]), float(by[i])) for i in order]


class VideoCanvas(QLabel):
    clicked = Signal(float, float, object)
    double_clicked = Signal(float, float, object)
    dragged = Signal(float, float, object)
    released = Signal(float, float, object)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(500, 500)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._image_shape = (1, 1)
        self._pixmap_rect = QRect()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        mapped = self.map_event_to_image(event)
        if mapped is None:
            return
        x, y = mapped
        self.clicked.emit(float(x), float(y), event.button())

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[override]
        mapped = self.map_event_to_image(event)
        if mapped is None:
            return
        x, y = mapped
        self.double_clicked.emit(float(x), float(y), event.button())

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        mapped = self.map_event_to_image(event)
        if mapped is None:
            return
        x, y = mapped
        self.dragged.emit(float(x), float(y), event.buttons())

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        mapped = self.map_event_to_image(event)
        if mapped is None:
            return
        x, y = mapped
        self.released.emit(float(x), float(y), event.button())

    def map_event_to_image(self, event) -> tuple[float, float] | None:
        if self._pixmap_rect.width() <= 0 or self._pixmap_rect.height() <= 0:
            return None
        point = event.position().toPoint()
        if not self._pixmap_rect.contains(point):
            return None
        x = (point.x() - self._pixmap_rect.left()) / self._pixmap_rect.width() * self._image_shape[1]
        y = (point.y() - self._pixmap_rect.top()) / self._pixmap_rect.height() * self._image_shape[0]
        return float(x), float(y)

    def set_rendered_pixmap(self, pixmap: QPixmap, image_shape: tuple[int, int]) -> None:
        self._image_shape = image_shape
        scaled = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        left = (self.width() - scaled.width()) // 2
        top = (self.height() - scaled.height()) // 2
        self._pixmap_rect = QRect(left, top, scaled.width(), scaled.height())
        self.setPixmap(scaled)


class TraceCanvas(FigureCanvas):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(5.0, 3.6), tight_layout=True)
        super().__init__(self.figure)
        self.setMinimumHeight(300)

    def plot_empty(self, text: str) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        self.draw_idle()

    def plot_traces(self, title: str, f: np.ndarray, fneu: np.ndarray | None, fcorr: np.ndarray, dff: np.ndarray) -> None:
        self.figure.clear()
        axes = self.figure.subplots(3, 1, sharex=True)
        x = np.arange(len(f))
        axes[0].plot(x, f, color="#2b6cb0", linewidth=0.9, label="F")
        if fneu is not None and len(fneu) == len(f):
            axes[0].plot(x, fneu, color="#718096", linewidth=0.8, label="Fneu")
        axes[0].legend(loc="upper right", fontsize=7)
        axes[0].set_ylabel("raw")
        axes[0].set_title(title, fontsize=9)

        axes[1].plot(x, fcorr, color="#2f855a", linewidth=0.9)
        axes[1].set_ylabel("F-0.7Fneu")

        axes[2].plot(x, dff, color="#c05621", linewidth=0.9)
        axes[2].axhline(0, color="#888888", linewidth=0.6)
        axes[2].set_ylabel("dF/F")
        axes[2].set_xlabel("frame")
        for ax in axes:
            ax.grid(True, linewidth=0.35, alpha=0.35)
        self.draw_idle()


class CurationWindow(QMainWindow):
    def __init__(
        self,
        paths: TrialPaths,
        data_root: Path,
        movie_kind: str,
        neuropil_coeff: float,
        f0_percentile: float,
        f0_eps: float,
    ) -> None:
        super().__init__()
        self.paths = paths
        self.data_root = data_root
        self.movie_kind = movie_kind
        self.trial_ids = discover_trial_ids(data_root, movie_kind)
        if paths.trial_id not in self.trial_ids:
            self.trial_ids.insert(0, paths.trial_id)
        self.neuropil_coeff = float(neuropil_coeff)
        self.f0_percentile = float(f0_percentile)
        self.f0_eps = float(f0_eps)
        self.stat = np.array([], dtype=object)
        self.roi_cache: list[dict] = []
        self.iscell = np.zeros((0, 2), dtype=np.float32)
        self.F = None
        self.Fneu = None
        self.suite2p_loaded = False
        self.suite2p_ref_flags = np.zeros((0,), dtype=bool)
        self.movie = movie_as_tyx(paths.movie_path)
        self.trace_movie_path = self.default_trace_movie_path(paths)
        self.trace_movie = movie_as_tyx(self.trace_movie_path)
        self.frame_index = 0
        self.selected_roi: int | None = None
        self.selected_manual_roi: int | None = None
        self.show_suite2p_refs = False
        self.selected_suite2p_refs: set[int] = set()
        self.deleted_existing: set[int] = set()
        self.added_rois: list[dict] = []
        self.current_polygon: list[tuple[float, float]] = []
        self.ellipse_start: tuple[float, float] | None = None
        self.ellipse_current: tuple[float, float] | None = None
        self.freehand_drawing = False
        self.undo_stack: list[dict] = []
        self.dirty = False
        self.low_pct = 1.0
        self.high_pct = 99.0
        self.playing = False

        self.setWindowTitle(f"ROI curation - {paths.trial_id}")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.left_canvas = VideoCanvas()
        self.right_canvas = VideoCanvas()
        self.left_canvas.clicked.connect(self.handle_overlay_click)
        self.left_canvas.double_clicked.connect(self.handle_overlay_double_click)
        self.right_canvas.clicked.connect(self.handle_right_click)
        self.right_canvas.dragged.connect(self.handle_right_drag)
        self.right_canvas.released.connect(self.handle_right_release)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(max(self.movie.shape[0] - 1, 0))
        self.frame_slider.sliderPressed.connect(self.pause_for_manual_frame_scrub)
        self.frame_slider.valueChanged.connect(self.set_frame)

        self.frame_spin = QSpinBox()
        self.frame_spin.setMinimum(0)
        self.frame_spin.setMaximum(max(self.movie.shape[0] - 1, 0))
        self.frame_spin.valueChanged.connect(self.set_frame)

        self.play_timer = QTimer(self)
        self.play_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.play_timer.timeout.connect(self.advance_frame)

        self.play_btn = QPushButton("Play")
        self.play_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.play_btn.clicked.connect(self.toggle_playback)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(10)
        self.fps_spin.setSuffix(" fps")
        self.fps_spin.valueChanged.connect(self.update_play_timer_interval)
        self.update_play_timer_interval()

        self.black_spin = QDoubleSpinBox()
        self.black_spin.setRange(0.0, 99.8)
        self.black_spin.setDecimals(1)
        self.black_spin.setSingleStep(0.5)
        self.black_spin.setSuffix(" %")
        self.black_spin.setValue(self.low_pct)
        self.black_spin.valueChanged.connect(self.update_display_contrast)

        self.white_spin = QDoubleSpinBox()
        self.white_spin.setRange(0.2, 100.0)
        self.white_spin.setDecimals(1)
        self.white_spin.setSingleStep(0.5)
        self.white_spin.setSuffix(" %")
        self.white_spin.setValue(self.high_pct)
        self.white_spin.valueChanged.connect(self.update_display_contrast)

        reset_display_btn = QPushButton("Reset display")
        reset_display_btn.clicked.connect(self.reset_display_contrast)

        self.roi_list = QListWidget()
        self.roi_list.currentRowChanged.connect(self.select_roi_from_list)

        self.movie_kind_combo = QComboBox()
        self.movie_kind_combo.addItem("raw (01 converted TIFF)", "raw")
        self.movie_kind_combo.addItem("motion corrected (03)", "corrected")
        self.movie_kind_combo.addItem("spatial high-pass (04)", "spatial-highpass")
        combo_index = self.movie_kind_combo.findData(movie_kind)
        if combo_index >= 0:
            self.movie_kind_combo.setCurrentIndex(combo_index)
        self.movie_kind_combo.currentIndexChanged.connect(self.change_movie_kind)

        self.trial_list = QListWidget()
        self.trial_list.setMaximumHeight(130)
        for trial_id in self.trial_ids:
            item = QListWidgetItem(trial_id)
            self.trial_list.addItem(item)
            if trial_id == paths.trial_id:
                self.trial_list.setCurrentItem(item)

        self.show_rejected = QCheckBox("show rejected")
        self.show_rejected.setChecked(False)
        self.show_rejected.stateChanged.connect(lambda _: self.rebuild_and_refresh())

        self.show_suite2p = QCheckBox("show suite2p refs")
        self.show_suite2p.setChecked(False)
        self.show_suite2p.stateChanged.connect(self.toggle_suite2p_refs)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["select ROI", "draw freehand ROI", "draw ellipse ROI"])
        self.mode_combo.currentIndexChanged.connect(lambda _: self.refresh())

        keep_btn = QPushButton("Keep selected")
        keep_btn.clicked.connect(lambda: self.set_selected_state(1))
        reject_btn = QPushButton("Reject selected")
        reject_btn.clicked.connect(lambda: self.set_selected_state(0))
        delete_btn = QPushButton("Delete selected")
        delete_btn.clicked.connect(self.delete_selected)
        clear_suite2p_btn = QPushButton("Clear suite2p picks")
        clear_suite2p_btn.clicked.connect(self.clear_suite2p_picks)
        load_trial_btn = QPushButton("Load selected")
        load_trial_btn.clicked.connect(self.load_selected_trial)
        next_trial_btn = QPushButton("Next")
        next_trial_btn.clicked.connect(self.load_next_trial)
        undo_action_btn = QPushButton("Undo")
        undo_action_btn.clicked.connect(self.undo_last_action)
        undo_btn = QPushButton("Undo drawing point")
        undo_btn.clicked.connect(self.undo_polygon_point)
        finish_btn = QPushButton("Finish drawn ROI")
        finish_btn.clicked.connect(self.finish_polygon_roi)
        clear_btn = QPushButton("Clear drawing")
        clear_btn.clicked.connect(self.clear_polygon)
        save_btn = QPushButton("Save manual curation")
        save_btn.clicked.connect(lambda: self.save_outputs())

        self.trace_canvas = TraceCanvas()

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        controls = QVBoxLayout()
        controls.addWidget(QLabel("Movie source"))
        controls.addWidget(self.movie_kind_combo)
        controls.addWidget(QLabel("Files"))
        controls.addWidget(self.trial_list)
        controls.addWidget(load_trial_btn)
        controls.addWidget(next_trial_btn)
        controls.addSpacing(10)
        controls.addWidget(QLabel("Display"))
        display_row = QHBoxLayout()
        display_row.addWidget(QLabel("black"))
        display_row.addWidget(self.black_spin)
        display_row.addWidget(QLabel("white"))
        display_row.addWidget(self.white_spin)
        controls.addLayout(display_row)
        controls.addWidget(reset_display_btn)
        controls.addSpacing(10)
        controls.addWidget(QLabel("ROI list"))
        controls.addWidget(self.roi_list)
        controls.addWidget(self.show_rejected)
        controls.addWidget(self.show_suite2p)
        controls.addWidget(keep_btn)
        controls.addWidget(reject_btn)
        controls.addWidget(delete_btn)
        controls.addWidget(clear_suite2p_btn)
        controls.addWidget(undo_action_btn)
        controls.addSpacing(10)
        controls.addWidget(QLabel("right image mode"))
        controls.addWidget(self.mode_combo)
        controls.addWidget(undo_btn)
        controls.addWidget(finish_btn)
        controls.addWidget(clear_btn)
        controls.addSpacing(10)
        controls.addWidget(QLabel("Trace"))
        controls.addWidget(self.trace_canvas)
        controls.addSpacing(10)
        controls.addWidget(save_btn)

        controls_widget = QWidget()
        controls_widget.setLayout(controls)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls_widget)
        controls_scroll.setMinimumWidth(360)

        image_layout = QHBoxLayout()
        image_layout.addWidget(self.left_canvas, stretch=1)
        image_layout.addWidget(self.right_canvas, stretch=1)

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("frame"))
        slider_layout.addWidget(self.play_btn)
        slider_layout.addWidget(self.fps_spin)
        slider_layout.addWidget(self.frame_slider)
        slider_layout.addWidget(self.frame_spin)

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.addLayout(image_layout)
        left_layout.addLayout(slider_layout)
        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addWidget(controls_scroll)

        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        self.load_saved_curation_if_present()
        self.populate_roi_list()
        self.refresh()
        self.update_trace_plot()

    def build_roi_cache(self) -> list[dict]:
        cache: list[dict] = []
        for roi in self.stat:
            if not isinstance(roi, dict):
                cache.append({"ypix": np.array([], dtype=np.int32), "xpix": np.array([], dtype=np.int32)})
                continue
            ypix = np.asarray(roi.get("ypix", []), dtype=np.int32)
            xpix = np.asarray(roi.get("xpix", []), dtype=np.int32)
            bypix, bxpix = suite2p_roi_boundary(ypix, xpix)
            cache.append(
                {
                    "ypix": ypix,
                    "xpix": xpix,
                    "boundary_ypix": np.asarray(bypix, dtype=np.int32),
                    "boundary_xpix": np.asarray(bxpix, dtype=np.int32),
                }
            )
        return cache

    @staticmethod
    def load_trace_array(path: Path | None, n_roi: int) -> np.ndarray | None:
        if path is None or not path.exists():
            return None
        arr = np.load(path, allow_pickle=True)
        if arr.ndim != 2 or arr.shape[0] != n_roi:
            return None
        return np.asarray(arr, dtype=np.float32)

    def populate_roi_list(self) -> None:
        current = self.selected_roi
        self.roi_list.blockSignals(True)
        self.roi_list.clear()
        for idx in sorted(self.selected_suite2p_refs):
            item = QListWidgetItem(f"suite2p {idx:04d} | selected")
            item.setData(Qt.ItemDataRole.UserRole, ("suite2p", idx))
            self.roi_list.addItem(item)
            if current == idx:
                self.roi_list.setCurrentItem(item)
        for idx, roi in enumerate(self.added_rois):
            status = roi.get("status", "accepted")
            if status == "rejected" and not self.show_rejected.isChecked():
                continue
            item = QListWidgetItem(f"manual {roi.get('manual_roi_id', idx + 1):04d} | {roi.get('roi_type', 'manual')} | {status}")
            item.setData(Qt.ItemDataRole.UserRole, ("manual", idx))
            self.roi_list.addItem(item)
            if self.selected_manual_roi == idx:
                self.roi_list.setCurrentItem(item)
        self.roi_list.blockSignals(False)

    def rebuild_and_refresh(self) -> None:
        self.populate_roi_list()
        self.refresh()

    def toggle_suite2p_refs(self) -> None:
        self.show_suite2p_refs = self.show_suite2p.isChecked()
        if self.show_suite2p_refs and not self.suite2p_loaded:
            self.load_suite2p_refs()
        self.refresh()

    def refresh_trial_list(self, selected_trial_id: str | None = None) -> None:
        self.trial_ids = discover_trial_ids(self.data_root, self.movie_kind)
        self.trial_list.blockSignals(True)
        self.trial_list.clear()
        for trial_id in self.trial_ids:
            item = QListWidgetItem(trial_id)
            self.trial_list.addItem(item)
            if selected_trial_id == trial_id:
                self.trial_list.setCurrentItem(item)
        if self.trial_list.currentRow() < 0 and self.trial_list.count() > 0:
            self.trial_list.setCurrentRow(0)
        self.trial_list.blockSignals(False)

    def change_movie_kind(self) -> None:
        new_kind = str(self.movie_kind_combo.currentData())
        if new_kind == self.movie_kind:
            return

        old_kind = self.movie_kind
        try:
            find_movie_path(self.data_root, self.paths.trial_id, new_kind)
        except Exception:
            current_trial_available = False
        else:
            current_trial_available = True

        if current_trial_available:
            self.switch_movie_source_for_current_trial(new_kind)
            return

        if not self.maybe_save_before_switch():
            old_index = self.movie_kind_combo.findData(old_kind)
            if old_index >= 0:
                self.movie_kind_combo.blockSignals(True)
                self.movie_kind_combo.setCurrentIndex(old_index)
                self.movie_kind_combo.blockSignals(False)
            return
        self.movie_kind = new_kind
        self.refresh_trial_list(selected_trial_id=self.paths.trial_id)
        if not self.trial_ids:
            QMessageBox.warning(self, "Movie source", f"No {self.movie_kind} movies found under:\n{self.data_root}")
            self.movie_kind = old_kind
            old_index = self.movie_kind_combo.findData(old_kind)
            if old_index >= 0:
                self.movie_kind_combo.blockSignals(True)
                self.movie_kind_combo.setCurrentIndex(old_index)
                self.movie_kind_combo.blockSignals(False)
            return
        target = self.paths.trial_id if self.paths.trial_id in self.trial_ids else self.trial_ids[0]
        self.load_trial_id(target, already_checked=True)

    def switch_movie_source_for_current_trial(self, new_kind: str) -> None:
        was_playing = self.playing
        if was_playing:
            self.pause_playback()
        try:
            paths = find_trial_paths(self.data_root, self.paths.trial_id, new_kind)
        except Exception as exc:
            QMessageBox.warning(self, "Movie source", str(exc))
            if was_playing:
                self.start_playback()
            return
        self.movie_kind = new_kind
        self.paths = paths
        self.movie = movie_as_tyx(paths.movie_path)
        self.frame_index = min(self.frame_index, max(self.movie.shape[0] - 1, 0))
        self.frame_slider.setMaximum(max(self.movie.shape[0] - 1, 0))
        self.frame_spin.setMaximum(max(self.movie.shape[0] - 1, 0))
        self.frame_slider.blockSignals(True)
        self.frame_spin.blockSignals(True)
        self.frame_slider.setValue(self.frame_index)
        self.frame_spin.setValue(self.frame_index)
        self.frame_slider.blockSignals(False)
        self.frame_spin.blockSignals(False)
        self.current_polygon = []
        self.ellipse_start = None
        self.ellipse_current = None
        self.freehand_drawing = False
        self.recompute_manual_roi_traces()
        self.refresh_trial_list(selected_trial_id=self.paths.trial_id)
        self.refresh()
        self.update_trace_plot()
        self.status.showMessage(f"Switched movie source to {new_kind}; ROI edits were kept unsaved")
        if was_playing:
            self.start_playback()

    def recompute_manual_roi_traces(self) -> None:
        refreshed: list[dict] = []
        for saved in self.added_rois:
            points = [(float(x), float(y)) for x, y in saved.get("points", [])]
            if len(points) < 3:
                continue
            mask = polygon_mask(points, self.movie.shape[-2:])
            if int(mask.sum()) < 3:
                continue
            roi = self.build_manual_roi(mask)
            roi.update({k: v for k, v in saved.items() if not k.startswith("_trace_")})
            roi["points"] = points
            refreshed.append(roi)
        self.added_rois = refreshed

    def default_trace_movie_path(self, paths: TrialPaths) -> Path:
        for kind in ("corrected", "raw"):
            try:
                return find_movie_path(self.data_root, paths.trial_id, kind)
            except Exception:
                continue
        return paths.movie_path

    def load_trace_movie_for_trial(self, paths: TrialPaths) -> None:
        self.trace_movie_path = self.default_trace_movie_path(paths)
        self.trace_movie = movie_as_tyx(self.trace_movie_path)

    def load_suite2p_refs(self) -> None:
        if self.paths.stat_path is None or not self.paths.stat_path.exists():
            self.show_suite2p.blockSignals(True)
            self.show_suite2p.setChecked(False)
            self.show_suite2p.blockSignals(False)
            self.show_suite2p_refs = False
            QMessageBox.warning(self, "suite2p refs", "No suite2p ROI output was found for this trial.")
            return
        self.stat = np.load(self.paths.stat_path, allow_pickle=True)
        self.roi_cache = self.build_roi_cache()
        self.iscell = np.zeros((len(self.stat), 2), dtype=np.float32)
        source_iscell = load_iscell(self.paths, len(self.stat))
        self.suite2p_ref_flags = source_iscell[:, 0].astype(bool)
        self.F = self.load_trace_array(self.paths.f_path, len(self.stat))
        self.Fneu = self.load_trace_array(self.paths.fneu_path, len(self.stat))
        self.suite2p_loaded = True
        self.status.showMessage(f"Loaded {len(self.stat)} suite2p reference ROIs")
        self.populate_roi_list()

    def load_saved_curation_if_present(self) -> None:
        out_dir = self.paths.output_dir
        additions_path = out_dir / f"{self.paths.trial_id}_manual_added_rois.json"
        roi_set_path = out_dir / f"{self.paths.trial_id}_manual_roi_set.json"
        selected_path = out_dir / f"{self.paths.trial_id}_selected_suite2p_indices.csv"
        deleted_path = out_dir / f"{self.paths.trial_id}_deleted_suite2p_indices.csv"
        loaded_parts: list[str] = []

        saved_roi_set = self.load_saved_roi_set(roi_set_path)
        saved_suite2p_points = self.saved_suite2p_points_by_id(saved_roi_set)
        selected_refs = read_index_csv(selected_path)
        deleted_refs = read_index_csv(deleted_path)
        valid_selected: set[int] = set()
        invalid_selected: set[int] = set()
        if selected_refs or deleted_refs:
            if not self.suite2p_loaded:
                self.load_suite2p_refs()
            for idx in selected_refs:
                if 0 <= idx < len(self.iscell):
                    saved_points = saved_suite2p_points.get(idx)
                    if saved_points is None or self.saved_suite2p_shape_matches_current(idx, saved_points):
                        valid_selected.add(idx)
                    else:
                        invalid_selected.add(idx)
                else:
                    invalid_selected.add(idx)
            valid_deleted = {idx for idx in deleted_refs if idx >= 0}
            self.selected_suite2p_refs = valid_selected
            self.deleted_existing = valid_deleted
            for idx in valid_selected:
                self.iscell[idx, 0] = 1.0
                self.iscell[idx, 1] = 1.0
            if valid_selected:
                self.show_suite2p_refs = True
                self.show_suite2p.blockSignals(True)
                self.show_suite2p.setChecked(True)
                self.show_suite2p.blockSignals(False)
            loaded_parts.append(f"{len(valid_selected)} suite2p pick(s)")

        if additions_path.exists():
            try:
                with additions_path.open("r", encoding="utf-8") as handle:
                    saved_rois = json.load(handle)
            except Exception as exc:
                QMessageBox.warning(self, "Load saved ROI", f"Could not load saved manual ROIs:\n{exc}")
                saved_rois = []
            self.added_rois = []
            for saved in saved_rois:
                points = [(float(x), float(y)) for x, y in saved.get("points", [])]
                if len(points) < 3:
                    continue
                mask = polygon_mask(points, self.movie.shape[-2:])
                if int(mask.sum()) < 3:
                    continue
                roi = self.build_manual_roi(mask)
                roi.update({k: v for k, v in saved.items() if not k.startswith("_trace_")})
                roi["points"] = points
                self.added_rois.append(roi)
            loaded_parts.append(f"{len(self.added_rois)} manual ROI(s)")

        imported = self.import_saved_suite2p_picks_as_manual(saved_roi_set, valid_selected | invalid_selected)
        if imported:
            loaded_parts.append(f"{imported} imported old suite2p ROI(s)")

        if loaded_parts:
            self.dirty = False
            self.undo_stack = []
            self.selected_roi = None
            self.selected_manual_roi = None
            self.status.showMessage("Loaded saved curation: " + ", ".join(loaded_parts))

    def load_saved_roi_set(self, roi_set_path: Path) -> list[dict]:
        if not roi_set_path.exists():
            return []
        try:
            with roi_set_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return []
        return payload if isinstance(payload, list) else []

    def saved_suite2p_points_by_id(self, saved_roi_set: list[dict]) -> dict[int, list[tuple[float, float]]]:
        points_by_id: dict[int, list[tuple[float, float]]] = {}
        for saved in saved_roi_set:
            if saved.get("roi_source") != "suite2p_reference":
                continue
            try:
                original_id = int(saved.get("suite2p_original_id"))
            except (TypeError, ValueError):
                continue
            points = [(float(x), float(y)) for x, y in saved.get("points", [])]
            if len(points) >= 3:
                points_by_id[original_id] = points
        return points_by_id

    def saved_suite2p_shape_matches_current(self, roi_idx: int, saved_points: list[tuple[float, float]]) -> bool:
        if roi_idx < 0 or roi_idx >= len(self.roi_cache):
            return False
        current = self.roi_cache[roi_idx]
        ypix = np.asarray(current.get("ypix", []), dtype=np.int32)
        xpix = np.asarray(current.get("xpix", []), dtype=np.int32)
        if ypix.size == 0 or xpix.size == 0:
            return False
        current_mask = np.zeros(self.movie.shape[-2:], dtype=bool)
        current_mask[ypix, xpix] = True
        saved_mask = polygon_mask(saved_points, self.movie.shape[-2:])
        intersection = int(np.logical_and(current_mask, saved_mask).sum())
        union = int(np.logical_or(current_mask, saved_mask).sum())
        if union == 0:
            return False
        return (intersection / union) >= 0.5

    def import_saved_suite2p_picks_as_manual(self, saved_roi_set: list[dict], selected_refs: set[int]) -> int:
        if not saved_roi_set:
            return 0
        existing_imports = {
            int(roi.get("suite2p_original_id"))
            for roi in self.added_rois
            if roi.get("roi_source") == "suite2p_reference_imported" and roi.get("suite2p_original_id") is not None
        }
        imported = 0
        for saved in saved_roi_set:
            if saved.get("roi_source") != "suite2p_reference":
                continue
            original_id = saved.get("suite2p_original_id")
            try:
                original_id_int = int(original_id)
            except (TypeError, ValueError):
                continue
            if original_id_int in self.selected_suite2p_refs or original_id_int in existing_imports:
                continue
            if selected_refs and original_id_int not in selected_refs:
                continue
            points = [(float(x), float(y)) for x, y in saved.get("points", [])]
            if len(points) < 3:
                continue
            mask = polygon_mask(points, self.movie.shape[-2:])
            if int(mask.sum()) < 3:
                continue
            roi = self.build_manual_roi(mask)
            roi.update(
                {
                    "manual_roi_id": self.next_manual_roi_id(),
                    "roi_type": "imported_suite2p",
                    "roi_source": "suite2p_reference_imported",
                    "status": "accepted",
                    "points": [[float(x), float(y)] for x, y in points],
                    "suite2p_original_id": original_id_int,
                }
            )
            self.added_rois.append(roi)
            existing_imports.add(original_id_int)
            imported += 1
        return imported

    def next_manual_roi_id(self) -> int:
        existing = [int(roi.get("manual_roi_id", 0)) for roi in self.added_rois]
        return max(existing, default=0) + 1

    def maybe_save_before_switch(self) -> bool:
        if not self.dirty:
            return True
        response = QMessageBox.question(
            self,
            "Unsaved ROI edits",
            "Current ROI edits are not saved. Save before switching?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if response == QMessageBox.StandardButton.Cancel:
            return False
        if response == QMessageBox.StandardButton.Save:
            self.save_outputs(show_message=False)
        return True

    def load_selected_trial(self) -> None:
        item = self.trial_list.currentItem()
        if item is None:
            return
        self.load_trial_id(item.text())

    def load_next_trial(self) -> None:
        if not self.trial_ids:
            return
        try:
            current_idx = self.trial_ids.index(self.paths.trial_id)
        except ValueError:
            current_idx = -1
        next_idx = (current_idx + 1) % len(self.trial_ids)
        self.trial_list.setCurrentRow(next_idx)
        self.load_trial_id(self.trial_ids[next_idx])

    def load_trial_id(self, trial_id: str, already_checked: bool = False) -> None:
        if trial_id == self.paths.trial_id:
            try:
                new_movie_path = find_movie_path(self.data_root, trial_id, self.movie_kind)
            except Exception:
                new_movie_path = None
            if new_movie_path == self.paths.movie_path:
                return
        if not already_checked and not self.maybe_save_before_switch():
            return
        was_playing = self.playing
        if was_playing:
            self.pause_playback()
        try:
            paths = find_trial_paths(self.data_root, trial_id, self.movie_kind)
        except Exception as exc:
            QMessageBox.warning(self, "Load trial", str(exc))
            if was_playing:
                self.start_playback()
            return
        self.paths = paths
        self.setWindowTitle(f"ROI curation - {paths.trial_id}")
        self.stat = np.array([], dtype=object)
        self.roi_cache = []
        self.iscell = np.zeros((0, 2), dtype=np.float32)
        self.F = None
        self.Fneu = None
        self.suite2p_loaded = False
        self.suite2p_ref_flags = np.zeros((0,), dtype=bool)
        self.movie = movie_as_tyx(paths.movie_path)
        self.load_trace_movie_for_trial(paths)
        self.frame_index = 0
        self.selected_roi = None
        self.selected_manual_roi = None
        self.show_suite2p_refs = False
        self.show_suite2p.blockSignals(True)
        self.show_suite2p.setChecked(False)
        self.show_suite2p.blockSignals(False)
        self.selected_suite2p_refs = set()
        self.deleted_existing = set()
        self.added_rois = []
        self.current_polygon = []
        self.ellipse_start = None
        self.ellipse_current = None
        self.freehand_drawing = False
        self.undo_stack = []
        self.dirty = False
        self.frame_slider.setMaximum(max(self.movie.shape[0] - 1, 0))
        self.frame_slider.setValue(0)
        self.frame_spin.setMaximum(max(self.movie.shape[0] - 1, 0))
        self.frame_spin.setValue(0)
        self.load_saved_curation_if_present()
        self.populate_roi_list()
        self.refresh()
        self.update_trace_plot()
        if was_playing:
            self.start_playback()

    def update_play_timer_interval(self) -> None:
        fps = max(int(self.fps_spin.value()), 1)
        self.play_timer.setInterval(max(1, int(round(1000.0 / fps))))

    def update_display_contrast(self) -> None:
        low = float(self.black_spin.value())
        high = float(self.white_spin.value())
        if high <= low:
            high = min(100.0, low + 0.2)
            self.white_spin.blockSignals(True)
            self.white_spin.setValue(high)
            self.white_spin.blockSignals(False)
        self.low_pct = low
        self.high_pct = high
        self.refresh()

    def reset_display_contrast(self) -> None:
        self.black_spin.blockSignals(True)
        self.white_spin.blockSignals(True)
        self.black_spin.setValue(1.0)
        self.white_spin.setValue(99.0)
        self.black_spin.blockSignals(False)
        self.white_spin.blockSignals(False)
        self.low_pct = 1.0
        self.high_pct = 99.0
        self.refresh()

    def toggle_playback(self) -> None:
        if self.playing:
            self.pause_playback()
        else:
            self.start_playback()

    def start_playback(self) -> None:
        self.playing = True
        self.play_btn.setText("Pause")
        self.update_play_timer_interval()
        self.play_timer.start()

    def pause_playback(self) -> None:
        self.playing = False
        self.play_btn.setText("Play")
        self.play_timer.stop()

    def pause_for_manual_frame_scrub(self) -> None:
        if self.playing:
            self.pause_playback()
            self.status.showMessage("Paused for manual frame browsing")

    def advance_frame(self) -> None:
        n_frame = int(self.movie.shape[0])
        if n_frame <= 1:
            return
        self.set_frame((self.frame_index + 1) % n_frame)

    def set_frame(self, value: int) -> None:
        max_frame = max(int(self.movie.shape[0]) - 1, 0)
        value = max(0, min(int(value), max_frame))
        if value == self.frame_index:
            return
        self.frame_index = value
        self.frame_slider.blockSignals(True)
        self.frame_spin.blockSignals(True)
        self.frame_slider.setValue(value)
        self.frame_spin.setValue(value)
        self.frame_slider.blockSignals(False)
        self.frame_spin.blockSignals(False)
        self.refresh()

    def select_roi_from_list(self, row: int) -> None:
        item = self.roi_list.item(row)
        if item is None:
            return
        kind, idx = item.data(Qt.ItemDataRole.UserRole)
        if kind == "suite2p":
            self.selected_roi = int(idx)
            self.selected_manual_roi = None
        else:
            self.selected_roi = None
            self.selected_manual_roi = int(idx)
        self.refresh()
        self.update_trace_plot()

    def handle_overlay_click(self, x: float, y: float, button: int) -> None:
        manual_idx = self.manual_roi_at(x, y)
        if manual_idx is not None:
            self.selected_roi = None
            self.selected_manual_roi = manual_idx
            if button == Qt.MouseButton.RightButton:
                self.set_selected_state(0)
                self.status.showMessage(f"Rejected manual ROI {self.added_rois[manual_idx].get('manual_roi_id')}")
            else:
                self.status.showMessage(f"Selected manual ROI {self.added_rois[manual_idx]['manual_roi_id']}")
                self.refresh()
                self.update_trace_plot()
            return
        if not self.show_suite2p_refs:
            return
        best_idx, best_dist = self.nearest_existing_roi(x, y)
        if button == Qt.MouseButton.RightButton:
            if best_idx is not None and best_dist <= 100:
                if self.remove_picked_suite2p_roi(best_idx):
                    self.status.showMessage(f"Removed suite2p ROI {best_idx} from picked refs")
            return
        if best_idx is not None and best_dist <= 100:
            self.selected_roi = best_idx
            self.selected_manual_roi = None
            self.status.showMessage(f"Viewing suite2p ROI {best_idx}. Double-click to pick/unpick.")
            self.refresh()
            self.update_trace_plot()
            return


    def handle_overlay_double_click(self, x: float, y: float, button: object) -> None:
        if not self.show_suite2p_refs or button != Qt.MouseButton.LeftButton:
            return
        best_idx, best_dist = self.nearest_existing_roi(x, y)
        if best_idx is None or best_dist > 100:
            return
        self.selected_roi = best_idx
        self.selected_manual_roi = None
        self.set_selected_state(1 if best_idx not in self.selected_suite2p_refs else 0)
        self.status.showMessage(f"Toggled suite2p ROI {best_idx}")

    def nearest_existing_roi(self, x: float, y: float) -> tuple[int | None, float]:
        best_idx = None
        best_dist = float("inf")
        for idx, roi in enumerate(self.roi_cache):
            if idx in self.deleted_existing:
                continue
            if idx < len(self.suite2p_ref_flags) and not self.suite2p_ref_flags[idx] and not self.show_rejected.isChecked():
                continue
            ypix = np.asarray(roi.get("ypix", []), dtype=float)
            xpix = np.asarray(roi.get("xpix", []), dtype=float)
            if ypix.size == 0:
                continue
            dist = float(np.min((xpix - x) ** 2 + (ypix - y) ** 2))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx, best_dist

    def nearest_selected_suite2p_roi(self, x: float, y: float) -> tuple[int | None, float]:
        best_idx = None
        best_dist = float("inf")
        for idx in sorted(self.selected_suite2p_refs):
            if idx in self.deleted_existing or idx >= len(self.roi_cache):
                continue
            roi = self.roi_cache[idx]
            ypix = np.asarray(roi.get("ypix", []), dtype=float)
            xpix = np.asarray(roi.get("xpix", []), dtype=float)
            if ypix.size == 0:
                continue
            dist = float(np.min((xpix - x) ** 2 + (ypix - y) ** 2))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx, best_dist

    def manual_roi_at(self, x: float, y: float) -> int | None:
        for idx in range(len(self.added_rois) - 1, -1, -1):
            roi = self.added_rois[idx]
            if roi.get("status", "accepted") == "rejected" and not self.show_rejected.isChecked():
                continue
            points = roi.get("points", [])
            if len(points) >= 3 and MplPath(points).contains_point((x, y)):
                return idx
        return None

    def handle_right_click(self, x: float, y: float, button: int) -> None:
        mode = self.mode_combo.currentText()
        if button == Qt.MouseButton.RightButton:
            manual_idx = self.manual_roi_at(x, y)
            if manual_idx is not None:
                manual_id = self.added_rois[manual_idx].get("manual_roi_id")
                self.delete_manual_roi_at(manual_idx)
                self.status.showMessage(f"Deleted manual ROI {manual_id}")
                return
            suite_idx, suite_dist = self.nearest_selected_suite2p_roi(x, y)
            if suite_idx is not None and suite_dist <= 100:
                if self.remove_picked_suite2p_roi(suite_idx):
                    self.status.showMessage(f"Removed suite2p ROI {suite_idx} from picked refs")
                return
        if mode == "draw freehand ROI":
            if button == Qt.MouseButton.LeftButton:
                self.current_polygon = [(float(x), float(y))]
                self.freehand_drawing = True
                self.status.showMessage("Freehand ROI started")
                self.refresh()
            elif button == Qt.MouseButton.RightButton:
                self.undo_polygon_point()
            return
        if mode == "draw ellipse ROI":
            if button == Qt.MouseButton.LeftButton:
                self.ellipse_start = (float(x), float(y))
                self.ellipse_current = (float(x), float(y))
                self.current_polygon = []
                self.status.showMessage("Ellipse ROI started")
                self.refresh()
            return
        if mode != "select ROI":
            return
        if mode == "select ROI":
            manual_idx = self.manual_roi_at(x, y)
            if manual_idx is not None:
                if button == Qt.MouseButton.RightButton:
                    manual_id = self.added_rois[manual_idx].get("manual_roi_id")
                    self.delete_manual_roi_at(manual_idx)
                    self.status.showMessage(f"Deleted manual ROI {manual_id}")
                else:
                    self.selected_roi = None
                    self.selected_manual_roi = manual_idx
                    self.refresh()
                    self.update_trace_plot()
                return
            suite_idx, suite_dist = self.nearest_selected_suite2p_roi(x, y)
            if suite_idx is not None and suite_dist <= 100:
                if button == Qt.MouseButton.RightButton:
                    if self.remove_picked_suite2p_roi(suite_idx):
                        self.status.showMessage(f"Removed suite2p ROI {suite_idx} from picked refs")
                else:
                    self.selected_roi = suite_idx
                    self.selected_manual_roi = None
                    self.refresh()
                    self.update_trace_plot()
            return

    def handle_right_drag(self, x: float, y: float, buttons: object) -> None:
        mode = self.mode_combo.currentText()
        if mode == "draw ellipse ROI" and self.ellipse_start is not None and (buttons & Qt.MouseButton.LeftButton):
            self.ellipse_current = (float(x), float(y))
            self.current_polygon = ellipse_points(self.ellipse_start, self.ellipse_current)
            self.refresh()
            return
        if mode != "draw freehand ROI":
            return
        if not self.freehand_drawing or not (buttons & Qt.MouseButton.LeftButton):
            return
        point = (float(x), float(y))
        if self.current_polygon:
            px, py = self.current_polygon[-1]
            if (px - point[0]) ** 2 + (py - point[1]) ** 2 < 0.75**2:
                return
        self.current_polygon.append(point)
        self.refresh()

    def handle_right_release(self, x: float, y: float, button: object) -> None:
        mode = self.mode_combo.currentText()
        if mode == "draw ellipse ROI" and button == Qt.MouseButton.LeftButton and self.ellipse_start is not None:
            self.ellipse_current = (float(x), float(y))
            self.current_polygon = ellipse_points(self.ellipse_start, self.ellipse_current)
            self.ellipse_start = None
            self.ellipse_current = None
            if len(self.current_polygon) >= 3:
                self.finish_polygon_roi()
            return
        if mode != "draw freehand ROI":
            return
        if button != Qt.MouseButton.LeftButton or not self.freehand_drawing:
            return
        self.freehand_drawing = False
        if self.current_polygon:
            self.current_polygon.append((float(x), float(y)))
        if len(self.current_polygon) >= 3:
            self.finish_polygon_roi()
        else:
            self.current_polygon = []
            self.refresh()

    def undo_polygon_point(self) -> None:
        if self.current_polygon:
            self.current_polygon.pop()
            self.refresh()

    def clear_polygon(self) -> None:
        if self.current_polygon:
            self.push_undo("clear-polygon")
        self.current_polygon = []
        self.ellipse_start = None
        self.ellipse_current = None
        self.freehand_drawing = False
        self.refresh()

    def finish_polygon_roi(self) -> None:
        if len(self.current_polygon) < 3:
            QMessageBox.warning(self, "Drawn ROI", "At least 3 points are needed.")
            return
        mask = polygon_mask(self.current_polygon, self.movie.shape[-2:])
        if int(mask.sum()) < 3:
            QMessageBox.warning(self, "Drawn ROI", "The ROI is too small.")
            return
        self.push_undo("add-manual")
        roi = self.build_manual_roi(mask)
        self.added_rois.append(roi)
        self.selected_roi = None
        self.selected_manual_roi = len(self.added_rois) - 1
        self.current_polygon = []
        self.dirty = True
        self.status.showMessage(f"Added {roi['roi_type']} ROI {roi['manual_roi_id']} with {roi['n_pixels']} pixels")
        self.refresh()
        self.update_trace_plot()

    def build_manual_roi(self, mask: np.ndarray) -> dict:
        ypix, xpix = np.nonzero(mask)
        annulus = annulus_mask(mask)
        f = self.mean_trace(mask)
        fneu = self.mean_trace(annulus) if np.any(annulus) else np.full_like(f, np.nan)
        if np.all(~np.isfinite(fneu)):
            fcorr = f.copy()
        else:
            fcorr = f - self.neuropil_coeff * np.nan_to_num(fneu, nan=0.0)
        dff, f0 = robust_dff(fcorr, self.f0_percentile, self.f0_eps)
        return {
            "manual_roi_id": len(self.added_rois) + 1,
            "roi_type": "ellipse" if self.mode_combo.currentText() == "draw ellipse ROI" else "freehand",
            "status": "accepted",
            "points": [[float(x), float(y)] for x, y in self.current_polygon],
            "frame_added": int(self.frame_index),
            "n_pixels": int(mask.sum()),
            "x_mean": float(np.mean(xpix)),
            "y_mean": float(np.mean(ypix)),
            "neuropil_coeff": float(self.neuropil_coeff),
            "f0_percentile": float(self.f0_percentile),
            "f0": float(f0),
            "trace_movie_path": str(self.trace_movie_path),
            "trace_summary": {
                "mean_F": float(np.nanmean(f)),
                "mean_Fneu": float(np.nanmean(fneu)),
                "mean_F_corrected": float(np.nanmean(fcorr)),
                "mean_dff": float(np.nanmean(dff)),
                "max_dff": float(np.nanmax(dff)),
            },
            "_trace_F": f.astype(float).tolist(),
            "_trace_Fneu": fneu.astype(float).tolist(),
            "_trace_F_corrected": fcorr.astype(float).tolist(),
            "_trace_dff": dff.astype(float).tolist(),
        }

    def mean_trace(self, mask: np.ndarray) -> np.ndarray:
        trace_movie = self.trace_movie
        if trace_movie.shape[-2:] != mask.shape:
            trace_movie = self.movie
        if not np.any(mask):
            return np.full((trace_movie.shape[0],), np.nan, dtype=np.float32)
        pixels = trace_movie[:, mask]
        return np.asarray(np.nanmean(pixels, axis=1), dtype=np.float32)

    def set_selected_state(self, state: int) -> None:
        if self.selected_manual_roi is not None and 0 <= self.selected_manual_roi < len(self.added_rois):
            self.push_undo("manual-state")
            self.added_rois[self.selected_manual_roi]["status"] = "accepted" if state else "rejected"
            self.dirty = True
            self.refresh()
            self.update_trace_plot()
            return
        if self.selected_roi is None or self.selected_roi in self.deleted_existing:
            return
        self.push_undo("suite2p-pick")
        if state:
            self.selected_suite2p_refs.add(int(self.selected_roi))
            self.iscell[self.selected_roi, 0] = 1.0
            self.iscell[self.selected_roi, 1] = 1.0
        else:
            self.selected_suite2p_refs.discard(int(self.selected_roi))
            self.iscell[self.selected_roi, 0] = 0.0
            self.iscell[self.selected_roi, 1] = 0.0
        self.dirty = True
        self.populate_roi_list()
        self.refresh()
        self.update_trace_plot()

    def remove_picked_suite2p_roi(self, roi_idx: int) -> bool:
        if roi_idx not in self.selected_suite2p_refs:
            return False
        self.push_undo("suite2p-unpick")
        self.selected_suite2p_refs.discard(int(roi_idx))
        if 0 <= roi_idx < len(self.iscell):
            self.iscell[roi_idx, 0] = 0.0
            self.iscell[roi_idx, 1] = 0.0
        if self.selected_roi == roi_idx:
            self.selected_roi = None
        self.dirty = True
        self.populate_roi_list()
        self.refresh()
        self.update_trace_plot()
        return True

    def delete_manual_roi_at(self, manual_idx: int) -> None:
        if manual_idx < 0 or manual_idx >= len(self.added_rois):
            return
        self.push_undo("delete-manual")
        removed = self.added_rois.pop(manual_idx)
        if self.selected_manual_roi == manual_idx:
            self.selected_manual_roi = None
        elif self.selected_manual_roi is not None and self.selected_manual_roi > manual_idx:
            self.selected_manual_roi -= 1
        self.dirty = True
        self.status.showMessage(f"Deleted manual ROI {removed.get('manual_roi_id')}")
        self.populate_roi_list()
        self.refresh()
        self.update_trace_plot()

    def delete_selected(self) -> None:
        if self.selected_manual_roi is not None and 0 <= self.selected_manual_roi < len(self.added_rois):
            self.delete_manual_roi_at(self.selected_manual_roi)
            return
        if self.selected_roi is None:
            return
        self.push_undo("delete-existing")
        deleted = int(self.selected_roi)
        self.deleted_existing.add(deleted)
        self.selected_suite2p_refs.discard(deleted)
        self.iscell[deleted, 0] = 0.0
        self.iscell[deleted, 1] = 0.0
        self.selected_roi = None
        self.selected_manual_roi = None
        self.dirty = True
        self.status.showMessage(f"Deleted suite2p ROI {deleted} from manual curation output")
        self.populate_roi_list()
        self.refresh()
        self.update_trace_plot()

    def clear_suite2p_picks(self) -> None:
        if not self.selected_suite2p_refs:
            self.status.showMessage("No suite2p picks to clear")
            return
        self.push_undo("clear-suite2p-picks")
        for idx in self.selected_suite2p_refs:
            self.iscell[idx, 0] = 0.0
            self.iscell[idx, 1] = 0.0
        self.selected_suite2p_refs = set()
        self.selected_roi = None
        self.dirty = True
        self.populate_roi_list()
        self.refresh()
        self.update_trace_plot()

    def push_undo(self, label: str) -> None:
        self.undo_stack.append(
            {
                "label": label,
                "iscell": self.iscell.copy(),
                "selected_suite2p_refs": set(self.selected_suite2p_refs),
                "deleted_existing": set(self.deleted_existing),
                "added_rois": copy.deepcopy(self.added_rois),
                "current_polygon": list(self.current_polygon),
                "ellipse_start": self.ellipse_start,
                "ellipse_current": self.ellipse_current,
                "selected_roi": self.selected_roi,
                "selected_manual_roi": self.selected_manual_roi,
            }
        )
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def undo_last_action(self) -> None:
        if not self.undo_stack:
            self.status.showMessage("Nothing to undo")
            return
        state = self.undo_stack.pop()
        self.iscell = state["iscell"]
        self.selected_suite2p_refs = state["selected_suite2p_refs"]
        self.deleted_existing = state["deleted_existing"]
        self.added_rois = state["added_rois"]
        self.current_polygon = state["current_polygon"]
        self.ellipse_start = state["ellipse_start"]
        self.ellipse_current = state["ellipse_current"]
        self.selected_roi = state["selected_roi"]
        self.selected_manual_roi = state["selected_manual_roi"]
        self.dirty = True
        self.status.showMessage(f"Undid {state['label']}")
        self.populate_roi_list()
        self.refresh()
        self.update_trace_plot()

    def update_trace_plot(self) -> None:
        if self.selected_manual_roi is not None and 0 <= self.selected_manual_roi < len(self.added_rois):
            roi = self.added_rois[self.selected_manual_roi]
            roi_type = roi.get("roi_type", "manual")
            self.trace_canvas.plot_traces(
                f"Manual {roi_type} ROI {roi['manual_roi_id']}",
                np.asarray(roi["_trace_F"], dtype=float),
                np.asarray(roi["_trace_Fneu"], dtype=float),
                np.asarray(roi["_trace_F_corrected"], dtype=float),
                np.asarray(roi["_trace_dff"], dtype=float),
            )
            return
        if self.selected_roi is None:
            self.trace_canvas.plot_empty("Select a ROI or draw a freehand/ellipse ROI")
            return
        if self.F is None:
            self.trace_canvas.plot_empty("F.npy is missing, so suite2p traces cannot be shown")
            return
        f = np.asarray(self.F[self.selected_roi], dtype=float)
        fneu = None if self.Fneu is None else np.asarray(self.Fneu[self.selected_roi], dtype=float)
        fcorr = f.copy() if fneu is None else f - self.neuropil_coeff * fneu
        dff, _ = robust_dff(fcorr, self.f0_percentile, self.f0_eps)
        self.trace_canvas.plot_traces(f"suite2p ROI {self.selected_roi}", f, fneu, fcorr, dff)

    def frame_pixmap(self, view: str) -> QPixmap:
        frame = normalize_frame(self.movie[self.frame_index], self.low_pct, self.high_pct)
        height, width = frame.shape
        qimg = QImage(frame.data, width, height, width, QImage.Format.Format_Grayscale8).copy()
        pixmap = QPixmap.fromImage(qimg)
        painter = QPainter(pixmap)
        if view == "reference":
            self.paint_suite2p_rois(painter, selected_only=False)
        elif view == "edit":
            self.paint_suite2p_rois(painter, selected_only=True)
        self.paint_added_rois(painter)
        self.paint_current_polygon(painter)
        painter.end()
        return pixmap

    def paint_suite2p_rois(self, painter: QPainter, selected_only: bool) -> None:
        if not self.show_suite2p_refs:
            return
        for idx, roi in enumerate(self.roi_cache):
            if idx in self.deleted_existing:
                continue
            if idx < len(self.suite2p_ref_flags) and not self.suite2p_ref_flags[idx] and not self.show_rejected.isChecked():
                continue
            keep = idx in self.selected_suite2p_refs
            if selected_only and not keep:
                continue
            ypix = np.asarray(roi.get("boundary_ypix", []), dtype=np.int32)
            xpix = np.asarray(roi.get("boundary_xpix", []), dtype=np.int32)
            if ypix.size == 0:
                continue
            color = QColor(0, 255, 80, 230) if keep else QColor(255, 128, 0, 120)
            if self.selected_roi == idx:
                color = QColor(255, 220, 0, 255)
            pen = QPen(color)
            pen.setWidth(1)
            painter.setPen(pen)
            for y, x in zip(ypix, xpix):
                painter.drawPoint(int(x), int(y))

    def paint_added_rois(self, painter: QPainter) -> None:
        pen = QPen(QColor(0, 180, 255, 230))
        pen.setWidth(1)
        painter.setPen(pen)
        for idx, roi in enumerate(self.added_rois):
            if roi.get("status", "accepted") == "rejected" and not self.show_rejected.isChecked():
                continue
            points = [QPointF(float(x), float(y)) for x, y in roi.get("points", [])]
            if len(points) < 2:
                continue
            if self.selected_manual_roi == idx:
                selected_pen = QPen(QColor(255, 220, 0, 255))
                selected_pen.setWidth(2)
                painter.setPen(selected_pen)
            elif roi.get("status", "accepted") == "rejected":
                rejected_pen = QPen(QColor(255, 60, 60, 160))
                rejected_pen.setWidth(1)
                painter.setPen(rejected_pen)
            else:
                painter.setPen(pen)
            for p0, p1 in zip(points, points[1:] + points[:1]):
                painter.drawLine(p0, p1)

    def paint_current_polygon(self, painter: QPainter) -> None:
        if not self.current_polygon:
            return
        pen = QPen(QColor(255, 120, 0, 220))
        pen.setWidth(1)
        painter.setPen(pen)
        points = [QPointF(float(x), float(y)) for x, y in self.current_polygon]
        for p0, p1 in zip(points, points[1:]):
            painter.drawLine(p0, p1)

    def refresh(self) -> None:
        shape = (self.movie.shape[-2], self.movie.shape[-1])
        self.left_canvas.set_rendered_pixmap(self.frame_pixmap(view="reference"), shape)
        self.right_canvas.set_rendered_pixmap(self.frame_pixmap(view="edit"), shape)
        kept = int(len(self.selected_suite2p_refs))
        self.status.showMessage(
            f"trial={self.paths.trial_id} | frame={self.frame_index}/{self.movie.shape[0]-1} | "
            f"suite2p picks={kept} | deleted refs={len(self.deleted_existing)} | "
            f"manual={len(self.added_rois)} | drawing points={len(self.current_polygon)}"
        )

    def roi_center_candidates(self) -> list[tuple[str, int, float, float]]:
        candidates: list[tuple[str, int, float, float]] = []
        if self.show_suite2p_refs:
            for idx, roi in enumerate(self.roi_cache):
                if idx in self.deleted_existing:
                    continue
                if idx < len(self.suite2p_ref_flags) and not self.suite2p_ref_flags[idx] and not self.show_rejected.isChecked():
                    continue
                xpix = np.asarray(roi.get("xpix", []), dtype=float)
                ypix = np.asarray(roi.get("ypix", []), dtype=float)
                if xpix.size:
                    candidates.append(("suite2p", idx, float(np.mean(xpix)), float(np.mean(ypix))))
        for idx, roi in enumerate(self.added_rois):
            if roi.get("status", "accepted") == "rejected" and not self.show_rejected.isChecked():
                continue
            candidates.append(("manual", idx, float(roi.get("x_mean", 0.0)), float(roi.get("y_mean", 0.0))))
        return candidates

    def current_center(self) -> tuple[float, float] | None:
        if self.selected_manual_roi is not None and 0 <= self.selected_manual_roi < len(self.added_rois):
            roi = self.added_rois[self.selected_manual_roi]
            return float(roi.get("x_mean", 0.0)), float(roi.get("y_mean", 0.0))
        if self.selected_roi is not None and 0 <= self.selected_roi < len(self.roi_cache):
            roi = self.roi_cache[self.selected_roi]
            xpix = np.asarray(roi.get("xpix", []), dtype=float)
            ypix = np.asarray(roi.get("ypix", []), dtype=float)
            if xpix.size:
                return float(np.mean(xpix)), float(np.mean(ypix))
        candidates = self.roi_center_candidates()
        if candidates:
            _, _, x, y = candidates[0]
            return x, y
        return None

    def move_selection(self, dx: int, dy: int) -> None:
        current = self.current_center()
        if current is None:
            return
        cx, cy = current
        best: tuple[float, str, int] | None = None
        for kind, idx, x, y in self.roi_center_candidates():
            if kind == "suite2p" and self.selected_roi == idx:
                continue
            if kind == "manual" and self.selected_manual_roi == idx:
                continue
            vx = x - cx
            vy = y - cy
            if dx < 0 and vx >= -1:
                continue
            if dx > 0 and vx <= 1:
                continue
            if dy < 0 and vy >= -1:
                continue
            if dy > 0 and vy <= 1:
                continue
            direction_penalty = abs(vy) if dx else abs(vx)
            forward_distance = abs(vx) if dx else abs(vy)
            score = forward_distance + 0.35 * direction_penalty
            if best is None or score < best[0]:
                best = (score, kind, idx)
        if best is None:
            return
        _, kind, idx = best
        if kind == "suite2p":
            self.selected_roi = idx
            self.selected_manual_roi = None
        else:
            self.selected_roi = None
            self.selected_manual_roi = idx
        self.refresh()
        self.update_trace_plot()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        key = event.key()
        modifiers = event.modifiers()
        if key == Qt.Key.Key_Space:
            self.toggle_playback()
            return
        if key == Qt.Key.Key_Left:
            if self.selected_roi is None and self.selected_manual_roi is None:
                self.set_frame(self.frame_index - 1)
            else:
                self.move_selection(-1, 0)
            return
        if key == Qt.Key.Key_Right:
            if self.selected_roi is None and self.selected_manual_roi is None:
                self.set_frame(self.frame_index + 1)
            else:
                self.move_selection(1, 0)
            return
        if key == Qt.Key.Key_Up:
            self.move_selection(0, -1)
            return
        if key == Qt.Key.Key_Down:
            self.move_selection(0, 1)
            return
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected()
            return
        if key == Qt.Key.Key_X:
            self.set_selected_state(0)
            return
        if key == Qt.Key.Key_K:
            self.set_selected_state(1)
            return
        if key == Qt.Key.Key_S:
            self.show_suite2p.setChecked(not self.show_suite2p.isChecked())
            return
        if key == Qt.Key.Key_Z and modifiers & Qt.KeyboardModifier.ControlModifier:
            self.undo_last_action()
            return
        if key == Qt.Key.Key_U:
            self.undo_last_action()
            return
        if key == Qt.Key.Key_Escape:
            self.undo_polygon_point()
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.mode_combo.currentText() in {"draw freehand ROI", "draw ellipse ROI"}:
            self.finish_polygon_roi()
            return
        super().keyPressEvent(event)

    def save_outputs(self, show_message: bool = True) -> None:
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        iscell_path = self.paths.output_dir / f"{self.paths.trial_id}_iscell_manual.npy"
        kept_path = self.paths.output_dir / f"{self.paths.trial_id}_selected_suite2p_indices.csv"
        deleted_path = self.paths.output_dir / f"{self.paths.trial_id}_deleted_suite2p_indices.csv"
        additions_path = self.paths.output_dir / f"{self.paths.trial_id}_manual_added_rois.json"
        new_roi_set_path = self.paths.output_dir / f"{self.paths.trial_id}_manual_roi_set.json"
        trace_path = self.paths.output_dir / f"{self.paths.trial_id}_manual_added_roi_traces.csv"
        suite2p_compat_dir = self.paths.output_dir / "suite2p_compatible" / "plane0"
        fiji_roi_zip_path = self.paths.output_dir / "RoiSet.zip"
        summary_path = self.paths.output_dir / f"{self.paths.trial_id}_manual_curation_summary.json"
        suite2p_compat_dir.mkdir(parents=True, exist_ok=True)
        iscell_to_save = np.zeros_like(self.iscell, dtype=np.float32)
        for idx in self.selected_suite2p_refs:
            iscell_to_save[idx, 0] = 1.0
            iscell_to_save[idx, 1] = 1.0
        if self.deleted_existing:
            deleted_arr = np.asarray(sorted(self.deleted_existing), dtype=int)
            iscell_to_save[deleted_arr, 0] = 0.0
            iscell_to_save[deleted_arr, 1] = 0.0
        np.save(iscell_path, iscell_to_save)

        kept_indices = np.asarray(sorted(self.selected_suite2p_refs), dtype=int)
        np.savetxt(kept_path, kept_indices, fmt="%d", delimiter=",", header="suite2p_original_id", comments="")
        np.savetxt(
            deleted_path,
            np.asarray(sorted(self.deleted_existing), dtype=int),
            fmt="%d",
            delimiter=",",
            header="suite2p_original_id",
            comments="",
        )

        serializable_rois = []
        new_roi_set = []
        suite2p_stat: list[dict] = []
        suite2p_f_rows: list[np.ndarray] = []
        suite2p_fneu_rows: list[np.ndarray] = []
        fiji_rois: list[tuple[str, list[tuple[float, float]]]] = []
        for idx in sorted(self.selected_suite2p_refs):
            roi = self.roi_cache[idx]
            xpix = np.asarray(roi.get("xpix", []), dtype=int)
            ypix = np.asarray(roi.get("ypix", []), dtype=int)
            lam = np.ones(len(xpix), dtype=np.float32)
            suite2p_stat.append(
                {
                    "ypix": ypix.astype(np.int32),
                    "xpix": xpix.astype(np.int32),
                    "lam": lam,
                    "med": np.array([float(np.mean(ypix)), float(np.mean(xpix))], dtype=np.float32) if xpix.size else np.array([np.nan, np.nan]),
                    "npix": int(len(xpix)),
                    "overlap": np.zeros(len(xpix), dtype=bool),
                    "suite2p_original_id": int(idx),
                }
            )
            if self.F is not None and idx < self.F.shape[0]:
                suite2p_f_rows.append(np.asarray(self.F[idx], dtype=np.float32))
            if self.Fneu is not None and idx < self.Fneu.shape[0]:
                suite2p_fneu_rows.append(np.asarray(self.Fneu[idx], dtype=np.float32))
            points = ordered_boundary_points(ypix, xpix)
            if len(points) >= 3:
                fiji_rois.append((safe_roi_name("suite2p", int(idx)), points))
            new_roi_set.append(
                {
                    "roi_id": f"suite2p_{idx}",
                    "roi_source": "suite2p_reference",
                    "suite2p_original_id": int(idx),
                    "n_pixels": int(len(xpix)),
                    "x_mean": float(np.mean(xpix)) if xpix.size else None,
                    "y_mean": float(np.mean(ypix)) if ypix.size else None,
                    "points": [[float(x), float(y)] for x, y in points],
                }
            )
        trace_rows = []
        for roi in self.added_rois:
            clean = {k: v for k, v in roi.items() if not k.startswith("_trace_")}
            serializable_rois.append(clean)
            if roi.get("status", "accepted") == "accepted":
                points = [(float(x), float(y)) for x, y in roi.get("points", [])]
                if len(points) >= 3:
                    mask = polygon_mask(points, self.movie.shape[-2:])
                    suite2p_stat.append(stat_dict_from_mask(mask, int(roi["manual_roi_id"])))
                    fiji_rois.append((safe_roi_name("manual", int(roi["manual_roi_id"])), points))
                suite2p_f_rows.append(np.asarray(roi["_trace_F"], dtype=np.float32))
                suite2p_fneu_rows.append(np.asarray(roi["_trace_Fneu"], dtype=np.float32))
                new_roi_set.append(
                    {
                        "roi_id": f"manual_{roi['manual_roi_id']}",
                        "roi_source": "manual",
                        **clean,
                    }
                )
            n_frames = len(roi.get("_trace_F", []))
            for frame in range(n_frames):
                trace_rows.append(
                    [
                        int(roi["manual_roi_id"]),
                        frame,
                        roi["_trace_F"][frame],
                        roi["_trace_Fneu"][frame],
                        roi["_trace_F_corrected"][frame],
                        roi["_trace_dff"][frame],
                    ]
                )
        with additions_path.open("w", encoding="utf-8") as handle:
            json.dump(serializable_rois, handle, indent=2)
        with new_roi_set_path.open("w", encoding="utf-8") as handle:
            json.dump(new_roi_set, handle, indent=2)
        np.save(suite2p_compat_dir / "stat.npy", np.asarray(suite2p_stat, dtype=object))
        compat_iscell = np.ones((len(suite2p_stat), 2), dtype=np.float32)
        np.save(suite2p_compat_dir / "iscell.npy", compat_iscell)
        if suite2p_f_rows and len(suite2p_f_rows) == len(suite2p_stat):
            frame_counts = {len(row) for row in suite2p_f_rows}
            if len(frame_counts) == 1:
                np.save(suite2p_compat_dir / "F.npy", np.vstack(suite2p_f_rows).astype(np.float32))
        if suite2p_fneu_rows and len(suite2p_fneu_rows) == len(suite2p_stat):
            frame_counts = {len(row) for row in suite2p_fneu_rows}
            if len(frame_counts) == 1:
                np.save(suite2p_compat_dir / "Fneu.npy", np.vstack(suite2p_fneu_rows).astype(np.float32))
        with zipfile.ZipFile(fiji_roi_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, points in fiji_rois:
                zf.writestr(name, imagej_polygon_roi_bytes(points, name))
        if trace_rows:
            with trace_path.open("w", encoding="utf-8") as handle:
                handle.write("manual_roi_id,frame,F,Fneu,F_corrected,dff\n")
                for row in trace_rows:
                    handle.write(",".join(str(value) for value in row) + "\n")
        summary = {
            "trial_id": self.paths.trial_id,
            "display_movie_path": str(self.paths.movie_path),
            "manual_trace_movie_path": str(self.trace_movie_path),
            "source_iscell": str(self.paths.curated_iscell_path or self.paths.iscell_path),
            "manual_iscell_path": str(iscell_path),
            "selected_suite2p_indices_path": str(kept_path),
            "deleted_suite2p_indices_path": str(deleted_path),
            "manual_added_rois_path": str(additions_path),
            "manual_roi_set_path": str(new_roi_set_path),
            "suite2p_compatible_dir": str(suite2p_compat_dir),
            "fiji_roiset_zip_path": str(fiji_roi_zip_path),
            "manual_added_roi_traces_path": str(trace_path) if trace_rows else None,
            "n_suite2p_roi": int(len(self.iscell)),
            "n_selected_suite2p_reference_roi": int(len(kept_indices)),
            "n_deleted_suite2p_reference_roi": int(len(self.deleted_existing)),
            "n_manual_added_roi": int(len(self.added_rois)),
            "n_manual_added_accepted_roi": int(sum(roi.get("status", "accepted") == "accepted" for roi in self.added_rois)),
            "n_manual_added_rejected_roi": int(sum(roi.get("status", "accepted") == "rejected" for roi in self.added_rois)),
            "n_new_roi_set": int(len(new_roi_set)),
            "n_suite2p_compatible_roi": int(len(suite2p_stat)),
            "n_fiji_roi": int(len(fiji_rois)),
            "neuropil_coeff": float(self.neuropil_coeff),
            "f0_percentile": float(self.f0_percentile),
            "note": (
                "The saved manual ROI set contains accepted manual ROIs plus selected "
                "suite2p reference ROIs only. Suite2p source files are not edited in place. "
                "Manual ROI traces use the motion-corrected movie when available, while "
                "brightness/contrast controls affect display only."
            ),
        }
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        self.dirty = False
        if show_message:
            QMessageBox.information(self, "Saved", f"Saved manual curation to:\n{self.paths.output_dir}")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.maybe_save_before_switch():
            event.accept()
        else:
            event.ignore()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open a local ROI curation GUI.")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--trial-id", help="Trial folder name. Defaults to the first available suite2p trial.")
    parser.add_argument("--movie-kind", choices=("raw", "corrected", "spatial-highpass"), default="corrected")
    parser.add_argument("--neuropil-coeff", type=float, default=0.7)
    parser.add_argument("--f0-percentile", type=float, default=10.0)
    parser.add_argument("--f0-eps", type=float, default=1e-6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app = QApplication(sys.argv[:1])
    data_root = args.data_root.expanduser().resolve() if args.data_root else None
    if data_root is None:
        selected = QFileDialog.getExistingDirectory(None, "Select data root")
        if not selected:
            return 1
        data_root = Path(selected).expanduser().resolve()
    trial_id = args.trial_id
    if trial_id is None:
        trial_ids = discover_trial_ids(data_root, args.movie_kind)
        if not trial_ids:
            QMessageBox.critical(None, "ROI curation", f"No movies found under:\n{data_root}")
            return 1
        trial_id = trial_ids[0]
    paths = find_trial_paths(data_root, trial_id, args.movie_kind)
    window = CurationWindow(
        paths,
        data_root=data_root,
        movie_kind=args.movie_kind,
        neuropil_coeff=args.neuropil_coeff,
        f0_percentile=args.f0_percentile,
        f0_eps=args.f0_eps,
    )
    window.resize(1550, 900)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
