#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desktop GUI for manual ROI curation after suite2p.

The GUI is intentionally conservative:
- it does not edit suite2p outputs in place
- selected suite2p candidates and drawn ROIs are saved as a manual ROI set
- temporary removals are kept only in the current GUI session
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
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
from PySide6.QtCore import QEvent, QItemSelectionModel, QPoint, QPointF, QRect, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


STEP_NAME = "05e_roi_manual_curation"
CELLPOSE_STEP_NAME = "05_cellpose_roi_segmentation"


@dataclass
class TrialPaths:
    trial_id: str
    rel_parent: Path
    suite2p_dir: Path | None
    plane0_dir: Path | None
    stat_path: Path | None
    iscell_path: Path | None
    f_path: Path | None
    fneu_path: Path | None
    movie_path: Path
    output_dir: Path


MOVIE_PATTERNS = {
    "raw": "*_Max_Proj.tif",
    "corrected": "*_corrected_movie.tif",
    "spatial-highpass": "*_spatial_highpass_movie.tif",
}

SETTINGS_ORG = "calcium-imaging-pipeline"
SETTINGS_APP = "manual-roi-curation"


def step_root_for_movie_kind(data_root: Path, movie_kind: str) -> Path:
    if movie_kind == "raw":
        return data_root / "01_oir_to_tif"
    if movie_kind == "spatial-highpass":
        return data_root / "04_spatial_highpass"
    return data_root / "03_motion_correct"


def manual_gui_settings() -> QSettings:
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


def last_trial_settings_key(data_root: Path, movie_kind: str) -> str:
    root_key = str(data_root.expanduser().resolve()).replace("/", "__").replace("\\", "__").replace(":", "_")
    return f"last_trial/{root_key}/{movie_kind}"


def save_last_trial_setting(data_root: Path, movie_kind: str, trial_id: str) -> None:
    settings = manual_gui_settings()
    settings.setValue(last_trial_settings_key(data_root, movie_kind), str(trial_id))
    settings.sync()


def load_last_trial_setting(data_root: Path, movie_kind: str, trial_ids: list[str]) -> str | None:
    saved = manual_gui_settings().value(last_trial_settings_key(data_root, movie_kind), "", type=str)
    if saved and saved in trial_ids:
        return str(saved)
    return None


def trial_rel_parent(trial_dir: Path, step_root: Path) -> Path:
    try:
        rel = trial_dir.parent.relative_to(step_root)
    except ValueError:
        return Path()
    return Path() if str(rel) == "." else rel


def manual_output_dir(data_root: Path, rel_parent: Path, trial_id: str) -> Path:
    if str(rel_parent) in ("", "."):
        return data_root / STEP_NAME / trial_id
    return data_root / STEP_NAME / rel_parent / trial_id


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
        rel_parent = trial_rel_parent(suite2p_dir, suite2p_root)
        iscell_path = plane0_dir / "iscell.npy"
        f_path = plane0_dir / "F.npy"
        fneu_path = plane0_dir / "Fneu.npy"
    else:
        if trial_id is None:
            trial_ids = discover_trial_ids(data_root, movie_kind)
            if not trial_ids:
                raise FileNotFoundError(
                    "No suite2p stat.npy or movie was found. Please pass --trial-id so the GUI can open a movie-only trial."
                )
            trial_id = trial_ids[0]
        plane0_dir = None
        suite2p_dir = None
        rel_parent = Path()
        iscell_path = None
        f_path = None
        fneu_path = None

    movie_path = find_movie_path(data_root, trial_id, movie_kind)
    if suite2p_dir is None:
        rel_parent = trial_rel_parent(movie_path.parent, step_root_for_movie_kind(data_root, movie_kind))
    return TrialPaths(
        trial_id=trial_id,
        rel_parent=rel_parent,
        suite2p_dir=suite2p_dir,
        plane0_dir=plane0_dir,
        stat_path=stat_path,
        iscell_path=iscell_path,
        f_path=f_path,
        fneu_path=fneu_path,
        movie_path=movie_path,
        output_dir=manual_output_dir(data_root, rel_parent, trial_id),
    )


def find_movie_path(data_root: Path, trial_id: str, movie_kind: str) -> Path:
    root = step_root_for_movie_kind(data_root, movie_kind)
    pattern = MOVIE_PATTERNS[movie_kind]
    matches = sorted((root / trial_id).glob(pattern))
    if matches:
        return matches[0]
    matches = sorted(path for path in root.rglob(pattern) if path.parent.name == trial_id)
    if matches:
        return matches[0]
    if movie_kind == "spatial-highpass":
        fallback = step_root_for_movie_kind(data_root, "corrected")
        matches = sorted((fallback / trial_id).glob(MOVIE_PATTERNS["corrected"]))
        if matches:
            return matches[0]
        matches = sorted(path for path in fallback.rglob(MOVIE_PATTERNS["corrected"]) if path.parent.name == trial_id)
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
        for movie in sorted(root.rglob(pattern)):
            trial_id = movie.parent.name
            if trial_id not in seen:
                seen.add(trial_id)
                trial_ids.append(trial_id)
    return trial_ids


def trial_display_label(data_root: Path, movie_kind: str, trial_id: str) -> str:
    try:
        movie_path = find_movie_path(data_root, trial_id, movie_kind)
    except Exception:
        return trial_id
    rel_parent = trial_rel_parent(movie_path.parent, step_root_for_movie_kind(data_root, movie_kind))
    if str(rel_parent) in ("", "."):
        return trial_id
    return f"{rel_parent} / {trial_id}"


def find_metadata_path(data_root: Path, trial_id: str, rel_parent: Path) -> Path | None:
    step_roots = (
        data_root / "03_motion_correct",
        data_root / "04_spatial_highpass",
        data_root / "01_oir_to_tif",
        data_root / "05_suite2p_roi_detection",
    )
    relative_trial = rel_parent / trial_id if str(rel_parent) not in ("", ".") else Path(trial_id)
    for root in step_roots:
        trial_dir = root / relative_trial
        matches = sorted(trial_dir.glob("*_metadata.json"))
        if matches:
            return matches[0]
    for root in step_roots:
        if not root.exists():
            continue
        matches = sorted(path for path in root.rglob("*_metadata.json") if path.parent.name == trial_id)
        if matches:
            return matches[0]
    return None


def nested_value(data: dict, keys: tuple[str, ...]) -> object | None:
    current: object = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def positive_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) and out > 0 else None


def fps_from_metadata_path(metadata_path: Path | None, default_fps: float = 2.0) -> tuple[float, str]:
    if metadata_path is None or not metadata_path.exists():
        return float(default_fps), "metadata missing"
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except Exception:
        return float(default_fps), "metadata unreadable"
    candidates = (
        nested_value(meta, ("temporal_calibration", "fps")),
        nested_value(meta, ("imaging", "fps")),
        nested_value(meta, ("acquisition", "fps")),
        meta.get("fps") if isinstance(meta, dict) else None,
        meta.get("frame_rate") if isinstance(meta, dict) else None,
    )
    for candidate in candidates:
        fps = positive_float(candidate)
        if fps is not None:
            return fps, str(metadata_path)
    intervals = (
        nested_value(meta, ("temporal_calibration", "frame_interval_sec")),
        nested_value(meta, ("temporal_calibration", "frame_interval")),
        meta.get("frame_interval_sec") if isinstance(meta, dict) else None,
        meta.get("frame_interval") if isinstance(meta, dict) else None,
    )
    for candidate in intervals:
        interval = positive_float(candidate)
        if interval is not None:
            return 1.0 / interval, str(metadata_path)
    return float(default_fps), "metadata fps missing"


def cellpose_plane_dir(data_root: Path, rel_parent: Path, trial_id: str) -> Path:
    if str(rel_parent) in ("", "."):
        return data_root / CELLPOSE_STEP_NAME / trial_id
    return data_root / CELLPOSE_STEP_NAME / rel_parent / trial_id


def load_iscell(paths: TrialPaths, n_roi: int) -> np.ndarray:
    if paths.iscell_path is None:
        out = np.zeros((n_roi, 2), dtype=np.float32)
        out[:, 1] = np.nan
        return out
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


def annulus_mask(
    roi_mask: np.ndarray,
    inner_px: int = 3,
    outer_px: int = 12,
    exclusion_mask: np.ndarray | None = None,
) -> np.ndarray:
    try:
        from scipy import ndimage as ndi
    except Exception:
        return np.zeros_like(roi_mask, dtype=bool)

    outer = ndi.binary_dilation(roi_mask, iterations=max(int(outer_px), 1))
    inner = ndi.binary_dilation(roi_mask, iterations=max(int(inner_px), 1))
    neuropil = outer & ~inner
    if exclusion_mask is not None and exclusion_mask.shape == roi_mask.shape:
        neuropil &= ~exclusion_mask
    return neuropil


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


def final_roi_sort_key(stat_entry: dict) -> tuple[float, float, int]:
    ypix = np.asarray(stat_entry.get("ypix", []), dtype=float)
    xpix = np.asarray(stat_entry.get("xpix", []), dtype=float)
    if ypix.size and xpix.size:
        return (float(np.mean(ypix)), float(np.mean(xpix)), -int(len(xpix)))
    med = stat_entry.get("med", [np.inf, np.inf])
    y_mean = float(med[0]) if len(med) > 0 and np.isfinite(med[0]) else np.inf
    x_mean = float(med[1]) if len(med) > 1 and np.isfinite(med[1]) else np.inf
    return (y_mean, x_mean, -int(stat_entry.get("npix", 0)))


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
    view_changed = Signal(float, float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(500, 500)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._image_shape = (1, 1)
        self._pixmap_rect = QRect()
        self._source_pixmap: QPixmap | None = None
        self._zoom = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 20.0
        self._center_x = 0.5
        self._center_y = 0.5
        self._syncing_view = False
        self._pan_enabled = False
        self._panning = False
        self._last_pan_point: QPoint | None = None

    def set_pan_enabled(self, enabled: bool) -> None:
        self._pan_enabled = bool(enabled)
        if self._pan_enabled:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.unsetCursor()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if self._should_start_pan(event):
            self._panning = True
            self._last_pan_point = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
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
        if self._panning and self._last_pan_point is not None:
            point = event.position().toPoint()
            delta = point - self._last_pan_point
            self._last_pan_point = point
            self.pan_by_screen_delta(delta.x(), delta.y(), emit=True)
            event.accept()
            return
        mapped = self.map_event_to_image(event)
        if mapped is None:
            return
        x, y = mapped
        self.dragged.emit(float(x), float(y), event.buttons())

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self._panning:
            self._panning = False
            self._last_pan_point = None
            self.setCursor(Qt.CursorShape.OpenHandCursor if self._pan_enabled else Qt.CursorShape.ArrowCursor)
            event.accept()
            return
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
        previous_shape = self._image_shape
        self._image_shape = image_shape
        self._source_pixmap = pixmap
        if previous_shape != image_shape:
            self.reset_view(emit=False)
        self._render_current_pixmap()

    def set_view(self, zoom: float, center_x: float, center_y: float, emit: bool = False) -> None:
        self._zoom = float(np.clip(zoom, self._min_zoom, self._max_zoom))
        self._center_x = float(center_x)
        self._center_y = float(center_y)
        self._clamp_center()
        self._render_current_pixmap()
        if emit and not self._syncing_view:
            self.view_changed.emit(self._zoom, self._center_x, self._center_y)

    def reset_view(self, emit: bool = True) -> None:
        height, width = self._image_shape
        self.set_view(1.0, width / 2.0, height / 2.0, emit=emit)

    def sync_view(self, zoom: float, center_x: float, center_y: float) -> None:
        self._syncing_view = True
        try:
            self.set_view(zoom, center_x, center_y, emit=False)
        finally:
            self._syncing_view = False

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._clamp_center()
        self._render_current_pixmap()

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        modifiers = event.modifiers()
        is_zoom = modifiers & Qt.KeyboardModifier.ControlModifier or modifiers & Qt.KeyboardModifier.MetaModifier
        angle_delta = event.angleDelta()
        pixel_delta = event.pixelDelta()
        if is_zoom:
            delta = angle_delta.y() or pixel_delta.y()
            if delta == 0:
                event.accept()
                return
            factor = float(np.clip(1.0015 ** delta, 0.5, 2.0))
            self.zoom_by(factor, event.position().toPoint())
            event.accept()
            return

        dx = pixel_delta.x() or angle_delta.x() / 4.0
        dy = pixel_delta.y() or angle_delta.y() / 4.0
        if self._zoom > 1.0 and (dx != 0 or dy != 0):
            self.pan_by_screen_delta(float(dx), float(dy), emit=True)
            event.accept()
            return
        event.ignore()

    def event(self, event) -> bool:  # type: ignore[override]
        if event.type() == QEvent.Type.NativeGesture:
            try:
                if event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                    factor = 1.0 + float(event.value())
                    if factor > 0:
                        self.zoom_by(factor, event.position().toPoint())
                        return True
            except AttributeError:
                pass
        return super().event(event)

    def zoom_by(self, factor: float, anchor: QPoint) -> None:
        if self._source_pixmap is None:
            return
        anchor_image = self._point_to_image(anchor)
        if anchor_image is None:
            height, width = self._image_shape
            anchor_image = (width / 2.0, height / 2.0)
            anchor = QPoint(self.width() // 2, self.height() // 2)
        new_zoom = float(np.clip(self._zoom * factor, self._min_zoom, self._max_zoom))
        scale = self._display_scale(new_zoom)
        if scale <= 0:
            return
        anchor_x, anchor_y = anchor_image
        center_x = (self.width() / 2.0 - (anchor.x() - anchor_x * scale)) / scale
        center_y = (self.height() / 2.0 - (anchor.y() - anchor_y * scale)) / scale
        self.set_view(new_zoom, center_x, center_y, emit=True)

    def pan_by_screen_delta(self, dx: float, dy: float, emit: bool = True) -> None:
        scale = self._display_scale()
        if scale <= 0 or self._zoom <= 1.0:
            return
        self.set_view(
            self._zoom,
            self._center_x - float(dx) / scale,
            self._center_y - float(dy) / scale,
            emit=emit,
        )

    def _should_start_pan(self, event) -> bool:
        if event.button() == Qt.MouseButton.MiddleButton:
            return True
        if self._pan_enabled and event.button() == Qt.MouseButton.LeftButton:
            return True
        return False

    def _point_to_image(self, point: QPoint) -> tuple[float, float] | None:
        if self._pixmap_rect.width() <= 0 or self._pixmap_rect.height() <= 0:
            return None
        if not self._pixmap_rect.contains(point):
            return None
        x = (point.x() - self._pixmap_rect.left()) / self._pixmap_rect.width() * self._image_shape[1]
        y = (point.y() - self._pixmap_rect.top()) / self._pixmap_rect.height() * self._image_shape[0]
        return float(x), float(y)

    def _display_scale(self, zoom: float | None = None) -> float:
        height, width = self._image_shape
        if height <= 0 or width <= 0 or self.width() <= 0 or self.height() <= 0:
            return 0.0
        fit = min(self.width() / width, self.height() / height)
        return fit * (self._zoom if zoom is None else zoom)

    def _clamp_center(self) -> None:
        height, width = self._image_shape
        scale = self._display_scale()
        if scale <= 0:
            return
        half_w = self.width() / (2.0 * scale)
        half_h = self.height() / (2.0 * scale)
        if half_w * 2.0 >= width:
            self._center_x = width / 2.0
        else:
            self._center_x = float(np.clip(self._center_x, half_w, width - half_w))
        if half_h * 2.0 >= height:
            self._center_y = height / 2.0
        else:
            self._center_y = float(np.clip(self._center_y, half_h, height - half_h))

    def _render_current_pixmap(self) -> None:
        if self._source_pixmap is None or self.width() <= 0 or self.height() <= 0:
            return
        self._clamp_center()
        scale = self._display_scale()
        height, width = self._image_shape
        display_w = max(1, int(round(width * scale)))
        display_h = max(1, int(round(height * scale)))
        left = int(round(self.width() / 2.0 - self._center_x * scale))
        top = int(round(self.height() / 2.0 - self._center_y * scale))
        self._pixmap_rect = QRect(left, top, display_w, display_h)

        canvas = QPixmap(self.size())
        canvas.fill(QColor(0, 0, 0))
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawPixmap(self._pixmap_rect, self._source_pixmap)
        painter.end()
        self.setPixmap(canvas)


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
        save_last_trial_setting(self.data_root, self.movie_kind, self.paths.trial_id)
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
        self.cellpose_loaded = False
        self.cellpose_cache: list[dict] = []
        self.movie = movie_as_tyx(paths.movie_path)
        self.trace_movie_path = self.default_trace_movie_path(paths)
        self.trace_movie = movie_as_tyx(self.trace_movie_path)
        self.acquisition_fps = 2.0
        self.acquisition_fps_source = ""
        self.refresh_acquisition_fps(paths)
        self.frame_index = 0
        self.selected_roi: int | None = None
        self.selected_manual_roi: int | None = None
        self.selected_removed_roi: int | None = None
        self.selection_box_start: tuple[float, float] | None = None
        self.selection_box_current: tuple[float, float] | None = None
        self.selection_box_target: str | None = None
        self.show_suite2p_refs = False
        self.show_cellpose_refs = False
        self.selected_candidate_suite2p_refs: set[int] = set()
        self.selected_candidate_cellpose_refs: set[int] = set()
        self.selected_suite2p_refs: set[int] = set()
        self.added_rois: list[dict] = []
        self.removed_rois: list[dict] = []
        self.current_polygon: list[tuple[float, float]] = []
        self.ellipse_start: tuple[float, float] | None = None
        self.ellipse_current: tuple[float, float] | None = None
        self.freehand_drawing = False
        self.undo_stack: list[dict] = []
        self.dirty = False
        self.low_pct = 1.0
        self.high_pct = 99.0
        self.playing = False
        self.image_order_swapped = False
        self._base_pixmap_cache: dict[tuple[int, float, float], QPixmap] = {}
        self._overlay_pixmap_cache: dict[tuple, QPixmap] = {}
        self._roi_overlay_revision = 0
        self._manual_traces_dirty = False
        self._trace_pause_notice_shown = False

        self.setWindowTitle(f"ROI curation - {paths.trial_id}")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.left_canvas = VideoCanvas()
        self.right_canvas = VideoCanvas()
        self.left_canvas.clicked.connect(self.handle_overlay_click)
        self.left_canvas.double_clicked.connect(self.handle_overlay_double_click)
        self.left_canvas.dragged.connect(self.handle_overlay_drag)
        self.left_canvas.released.connect(self.handle_overlay_release)
        self.right_canvas.clicked.connect(self.handle_right_click)
        self.right_canvas.dragged.connect(self.handle_right_drag)
        self.right_canvas.released.connect(self.handle_right_release)
        self.left_canvas.view_changed.connect(self.sync_canvas_view_from_left)
        self.right_canvas.view_changed.connect(self.sync_canvas_view_from_right)

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

        self.playback_mode_combo = QComboBox()
        self.playback_mode_combo.addItem("real-time x", "real-time")
        self.playback_mode_combo.addItem("display fps", "fps")
        self.playback_mode_combo.currentIndexChanged.connect(self.update_playback_controls)

        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.05, 100.0)
        self.speed_spin.setDecimals(2)
        self.speed_spin.setSingleStep(0.25)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setSuffix(" x")
        self.speed_spin.valueChanged.connect(self.update_play_timer_interval)

        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.1, 240.0)
        self.fps_spin.setDecimals(1)
        self.fps_spin.setSingleStep(1.0)
        self.fps_spin.setValue(10.0)
        self.fps_spin.setSuffix(" fps")
        self.fps_spin.valueChanged.connect(self.update_play_timer_interval)

        self.playback_info_label = QLabel("")
        self.update_playback_controls()

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

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItem("both images", "both")
        self.view_mode_combo.addItem("reference only", "reference")
        self.view_mode_combo.addItem("edit only", "edit")
        self.view_mode_combo.currentIndexChanged.connect(self.update_image_layout)

        self.swap_images_btn = QPushButton("Swap images")
        self.swap_images_btn.clicked.connect(self.swap_image_order)

        reset_zoom_btn = QPushButton("Reset zoom")
        reset_zoom_btn.clicked.connect(self.reset_canvas_zoom)
        clear_cache_btn = QPushButton("Clear image cache")
        clear_cache_btn.clicked.connect(self.clear_image_cache)

        self.pan_tool_check = QCheckBox("drag to pan")
        self.pan_tool_check.stateChanged.connect(self.toggle_pan_tool)

        self.render_scale_spin = QDoubleSpinBox()
        self.render_scale_spin.setRange(0.25, 1.0)
        self.render_scale_spin.setDecimals(2)
        self.render_scale_spin.setSingleStep(0.25)
        self.render_scale_spin.setSuffix(" x")
        self.render_scale_spin.setToolTip("Lower values render faster but blurrier; 1.00 keeps full display detail.")
        self.render_scale_spin.setValue(0.75 if platform.system() == "Linux" else 1.0)
        self.render_scale_spin.valueChanged.connect(self.update_render_scale)

        self.fast_play_single_view = QCheckBox("play left only")
        self.fast_play_single_view.setChecked(False)
        self.fast_play_no_overlays = QCheckBox("hide overlays while playing")
        self.fast_play_no_overlays.setChecked(False)
        self.fast_play_single_view.stateChanged.connect(lambda _: self.refresh())
        self.fast_play_no_overlays.stateChanged.connect(lambda _: self.rebuild_and_refresh())

        self.roi_table = QTableWidget(0, 4)
        self.roi_table.setHorizontalHeaderLabels(["source", "id", "type", "state"])
        self.roi_table.setMinimumHeight(150)
        self.roi_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.roi_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.roi_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.roi_table.verticalHeader().setVisible(False)
        self.roi_table.horizontalHeader().setStretchLastSection(True)
        self.roi_table.currentCellChanged.connect(self.select_roi_from_table)

        self.removed_roi_table = QTableWidget(0, 3)
        self.removed_roi_table.setHorizontalHeaderLabels(["source", "id", "type"])
        self.removed_roi_table.setMaximumHeight(120)
        self.removed_roi_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.removed_roi_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.removed_roi_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.removed_roi_table.verticalHeader().setVisible(False)
        self.removed_roi_table.horizontalHeader().setStretchLastSection(True)
        self.removed_roi_table.currentCellChanged.connect(self.select_removed_roi_from_table)

        self.movie_kind_combo = QComboBox()
        self.movie_kind_combo.addItem("raw (01 converted TIFF)", "raw")
        self.movie_kind_combo.addItem("motion corrected (03)", "corrected")
        self.movie_kind_combo.addItem("spatial high-pass (04)", "spatial-highpass")
        combo_index = self.movie_kind_combo.findData(movie_kind)
        if combo_index >= 0:
            self.movie_kind_combo.setCurrentIndex(combo_index)
        self.movie_kind_combo.currentIndexChanged.connect(self.change_movie_kind)

        self.date_combo = QComboBox()
        self.date_combo.setMaxVisibleItems(20)
        self.date_combo.currentIndexChanged.connect(self.refresh_trial_combo_for_date)

        self.trial_combo = QComboBox()
        self.trial_combo.setEditable(True)
        self.trial_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.trial_combo.setMaxVisibleItems(30)
        self.trial_combo.setMinimumContentsLength(28)
        self.refresh_trial_selectors(paths.trial_id)

        self.show_suite2p_iscell0 = QCheckBox("show suite2p iscell=0 refs")
        self.show_suite2p_iscell0.setChecked(False)
        self.show_suite2p_iscell0.stateChanged.connect(lambda _: self.rebuild_and_refresh())

        self.show_suite2p = QCheckBox("show suite2p refs")
        self.show_suite2p.setChecked(False)
        self.show_suite2p.stateChanged.connect(self.toggle_suite2p_refs)

        self.show_cellpose = QCheckBox("show cellpose refs")
        self.show_cellpose.setChecked(False)
        self.show_cellpose.stateChanged.connect(self.toggle_cellpose_refs)

        self.show_final_rois = QCheckBox("show final ROIs on right")
        self.show_final_rois.setChecked(True)
        self.show_final_rois.stateChanged.connect(lambda _: self.refresh())

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["select ROI", "draw freehand ROI", "draw ellipse ROI"])
        self.mode_combo.currentIndexChanged.connect(lambda _: self.refresh())

        keep_btn = QPushButton("Keep selected")
        keep_btn.clicked.connect(lambda: self.set_selected_state(1))
        keep_all_visible_btn = QPushButton("Keep all visible refs")
        keep_all_visible_btn.clicked.connect(self.keep_all_visible_refs)
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(lambda: self.set_selected_state(0))
        delete_btn = QPushButton("Delete selected")
        delete_btn.clicked.connect(self.delete_selected)
        restore_removed_btn = QPushButton("Restore removed")
        restore_removed_btn.clicked.connect(self.restore_removed_roi)
        clear_final_btn = QPushButton("Clear all final ROIs")
        clear_final_btn.clicked.connect(self.clear_final_rois)
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
        self.auto_trace_check = QCheckBox("auto trace")
        self.auto_trace_check.setChecked(False)
        self.auto_trace_check.setToolTip("If enabled, every ROI selection recomputes F/Fneu/dF/F immediately.")
        self.auto_trace_check.stateChanged.connect(lambda _: self.update_trace_plot(force=self.auto_trace_check.isChecked()))
        update_trace_btn = QPushButton("Update trace")
        update_trace_btn.clicked.connect(lambda: self.update_trace_plot(force=True))
        save_btn = QPushButton("Save manual curation")
        save_btn.clicked.connect(lambda: self.save_outputs())
        save_close_btn = QPushButton("Save and close")
        save_close_btn.clicked.connect(self.save_and_close)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.request_close)

        self.trace_canvas = TraceCanvas()

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        controls = QVBoxLayout()
        controls.addWidget(QLabel("Movie source"))
        controls.addWidget(self.movie_kind_combo)
        controls.addWidget(QLabel("Date"))
        controls.addWidget(self.date_combo)
        controls.addWidget(QLabel("Trial"))
        controls.addWidget(self.trial_combo)
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
        controls.addWidget(QLabel("View"))
        view_row = QHBoxLayout()
        view_row.addWidget(self.view_mode_combo)
        view_row.addWidget(self.swap_images_btn)
        controls.addLayout(view_row)
        controls.addWidget(self.pan_tool_check)
        controls.addWidget(reset_zoom_btn)
        controls.addWidget(clear_cache_btn)
        controls.addWidget(QLabel("Performance"))
        perf_row = QHBoxLayout()
        perf_row.addWidget(QLabel("render scale"))
        perf_row.addWidget(self.render_scale_spin)
        controls.addLayout(perf_row)
        controls.addWidget(self.fast_play_single_view)
        controls.addWidget(self.fast_play_no_overlays)
        controls.addSpacing(10)
        controls.addWidget(QLabel("ROI table"))
        controls.addWidget(self.roi_table)
        controls.addWidget(self.show_suite2p_iscell0)
        controls.addWidget(self.show_suite2p)
        controls.addWidget(self.show_cellpose)
        controls.addWidget(self.show_final_rois)
        controls.addWidget(keep_btn)
        controls.addWidget(keep_all_visible_btn)
        controls.addWidget(remove_btn)
        controls.addWidget(delete_btn)
        controls.addWidget(QLabel("Removed this session"))
        controls.addWidget(self.removed_roi_table)
        controls.addWidget(restore_removed_btn)
        controls.addWidget(clear_final_btn)
        controls.addWidget(undo_action_btn)
        controls.addSpacing(10)
        controls.addWidget(QLabel("right image mode"))
        controls.addWidget(self.mode_combo)
        controls.addWidget(undo_btn)
        controls.addWidget(finish_btn)
        controls.addWidget(clear_btn)
        controls.addSpacing(10)
        controls.addWidget(QLabel("Trace"))
        trace_row = QHBoxLayout()
        trace_row.addWidget(self.auto_trace_check)
        trace_row.addWidget(update_trace_btn)
        controls.addLayout(trace_row)
        controls.addWidget(self.trace_canvas)
        controls.addSpacing(10)
        controls.addWidget(save_btn)
        close_row = QHBoxLayout()
        close_row.addWidget(save_close_btn)
        close_row.addWidget(close_btn)
        controls.addLayout(close_row)

        controls_widget = QWidget()
        controls_widget.setLayout(controls)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls_widget)
        controls_scroll.setMinimumWidth(360)

        self.image_layout = QHBoxLayout()
        self.image_layout.addWidget(self.left_canvas, stretch=1)
        self.image_layout.addWidget(self.right_canvas, stretch=1)

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("frame"))
        slider_layout.addWidget(self.play_btn)
        slider_layout.addWidget(self.playback_mode_combo)
        slider_layout.addWidget(self.speed_spin)
        slider_layout.addWidget(self.fps_spin)
        slider_layout.addWidget(self.playback_info_label)
        slider_layout.addWidget(self.frame_slider)
        slider_layout.addWidget(self.frame_spin)

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.addLayout(self.image_layout)
        left_layout.addLayout(slider_layout)
        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addWidget(controls_scroll)

        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        self.load_saved_curation_if_present()
        self.populate_roi_list()
        self.update_image_layout()
        self.refresh()
        self.update_trace_plot()

    def sync_canvas_view_from_left(self, zoom: float, center_x: float, center_y: float) -> None:
        self.right_canvas.sync_view(zoom, center_x, center_y)

    def sync_canvas_view_from_right(self, zoom: float, center_x: float, center_y: float) -> None:
        self.left_canvas.sync_view(zoom, center_x, center_y)

    def invalidate_frame_cache(self) -> None:
        self._base_pixmap_cache.clear()

    def invalidate_overlay_cache(self) -> None:
        self._overlay_pixmap_cache.clear()
        self._roi_overlay_revision += 1

    def invalidate_all_image_caches(self) -> None:
        self.invalidate_frame_cache()
        self.invalidate_overlay_cache()

    def clear_image_cache(self) -> None:
        self.invalidate_all_image_caches()
        self.status.showMessage("Cleared image cache")
        self.refresh()

    def mark_manual_traces_dirty(self) -> None:
        self._manual_traces_dirty = True

    def ensure_manual_traces_current(self) -> None:
        if self._manual_traces_dirty:
            self.recompute_manual_roi_traces()

    def reset_canvas_zoom(self) -> None:
        self.left_canvas.reset_view(emit=False)
        self.right_canvas.reset_view(emit=False)
        self.status.showMessage("Zoom reset")

    def render_scale(self) -> float:
        return float(self.render_scale_spin.value()) if hasattr(self, "render_scale_spin") else 1.0

    def update_render_scale(self) -> None:
        self.invalidate_all_image_caches()
        self.refresh()

    def toggle_pan_tool(self) -> None:
        enabled = self.pan_tool_check.isChecked()
        self.left_canvas.set_pan_enabled(enabled)
        self.right_canvas.set_pan_enabled(enabled)
        if enabled:
            self.selection_box_start = None
            self.selection_box_current = None
            self.freehand_drawing = False
            self.ellipse_start = None
            self.ellipse_current = None
            self.status.showMessage("Drag to pan is on")
        else:
            self.status.showMessage("Drag to pan is off")
        self.refresh()

    def swap_image_order(self) -> None:
        self.image_order_swapped = not self.image_order_swapped
        self.update_image_layout()

    def update_image_layout(self, *_args) -> None:
        while self.image_layout.count():
            self.image_layout.takeAt(0)

        canvases = [self.left_canvas, self.right_canvas]
        if self.image_order_swapped:
            canvases = [self.right_canvas, self.left_canvas]
        for canvas in canvases:
            self.image_layout.addWidget(canvas, stretch=1)

        mode = self.view_mode_combo.currentData() if hasattr(self, "view_mode_combo") else "both"
        self.left_canvas.setVisible(mode in {"both", "reference"})
        self.right_canvas.setVisible(mode in {"both", "edit"})

        self.swap_images_btn.setText("Unswap images" if self.image_order_swapped else "Swap images")
        self.left_canvas._render_current_pixmap()
        self.right_canvas._render_current_pixmap()

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
    def normalize_candidate_cache(stat: np.ndarray) -> list[dict]:
        cache: list[dict] = []
        for roi in stat:
            if not isinstance(roi, dict):
                roi = dict(roi)
            ypix = np.asarray(roi.get("ypix", []), dtype=np.int32)
            xpix = np.asarray(roi.get("xpix", []), dtype=np.int32)
            bypix = np.asarray(roi.get("boundary_ypix", []), dtype=np.int32)
            bxpix = np.asarray(roi.get("boundary_xpix", []), dtype=np.int32)
            if bypix.size == 0 or bxpix.size == 0:
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
        selected_entries = set(self.selected_final_entries())
        current = self.selected_roi
        self.roi_table.blockSignals(True)
        self.roi_table.setRowCount(0)
        for idx in sorted(self.selected_suite2p_refs):
            row = self.add_roi_table_row("suite2p", idx, "suite2p", "final", ("suite2p", idx))
            if ("suite2p", idx) in selected_entries or (not selected_entries and current == idx):
                self.select_roi_table_row(row)
        for idx, roi in enumerate(self.added_rois):
            source_label = "cellpose" if roi.get("roi_source") == "cellpose" else "manual"
            row = self.add_roi_table_row(
                source_label,
                roi.get("manual_roi_id", idx + 1),
                roi.get("roi_type", "manual"),
                "final",
                ("manual", idx),
            )
            if ("manual", idx) in selected_entries or (not selected_entries and self.selected_manual_roi == idx):
                self.select_roi_table_row(row)
        self.roi_table.resizeColumnsToContents()
        self.roi_table.blockSignals(False)
        self.populate_removed_roi_table()

    def select_roi_table_row(self, row: int) -> None:
        self.roi_table.selectionModel().select(
            self.roi_table.model().index(row, 0),
            QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
        )

    def add_roi_table_row(self, source: str, roi_id: object, roi_type: str, state: str, data: tuple[str, int]) -> int:
        row = self.roi_table.rowCount()
        self.roi_table.insertRow(row)
        values = [str(source), str(roi_id), str(roi_type), str(state)]
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter if col in {1, 3} else Qt.AlignmentFlag.AlignLeft)
            if col == 0:
                item.setData(Qt.ItemDataRole.UserRole, data)
            self.roi_table.setItem(row, col, item)
        return row

    def selected_final_entries(self) -> list[tuple[str, int]]:
        entries: list[tuple[str, int]] = []
        for index in self.roi_table.selectionModel().selectedRows():
            item = self.roi_table.item(index.row(), 0)
            if item is None:
                continue
            kind, idx = item.data(Qt.ItemDataRole.UserRole)
            entries.append((str(kind), int(idx)))
        if entries:
            return entries
        if self.selected_manual_roi is not None:
            return [("manual", int(self.selected_manual_roi))]
        if self.selected_roi is not None and self.selected_roi in self.selected_suite2p_refs:
            return [("suite2p", int(self.selected_roi))]
        return []

    def selected_final_sets(self) -> tuple[set[int], set[int]]:
        suite2p: set[int] = set()
        manual: set[int] = set()
        for kind, idx in self.selected_final_entries():
            if kind == "suite2p":
                suite2p.add(idx)
            elif kind == "manual":
                manual.add(idx)
        return suite2p, manual

    @staticmethod
    def additive_selection_requested() -> bool:
        modifiers = QApplication.keyboardModifiers()
        return bool(
            modifiers
            & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
        )

    def suite2p_ref_visible(self, idx: int) -> bool:
        if idx in self.selected_suite2p_refs:
            return False
        if (
            idx < len(self.suite2p_ref_flags)
            and not self.suite2p_ref_flags[idx]
            and not self.show_suite2p_iscell0.isChecked()
        ):
            return False
        return True

    def cellpose_final_original_ids(self) -> set[int]:
        ids: set[int] = set()
        for roi in self.added_rois:
            if roi.get("roi_source") != "cellpose":
                continue
            try:
                ids.add(int(roi.get("cellpose_original_id")))
            except (TypeError, ValueError):
                continue
        return ids

    def cellpose_ref_visible(self, idx: int) -> bool:
        return int(idx) not in self.cellpose_final_original_ids()

    def valid_candidate_suite2p_selection(self) -> set[int]:
        return {
            int(idx)
            for idx in self.selected_candidate_suite2p_refs
            if 0 <= int(idx) < len(self.roi_cache) and self.suite2p_ref_visible(int(idx))
        }

    def valid_candidate_cellpose_selection(self) -> set[int]:
        return {
            int(idx)
            for idx in self.selected_candidate_cellpose_refs
            if 0 <= int(idx) < len(self.cellpose_cache) and self.cellpose_ref_visible(int(idx))
        }

    def clear_candidate_selection(self) -> None:
        self.selected_candidate_suite2p_refs = set()
        self.selected_candidate_cellpose_refs = set()

    @staticmethod
    def rect_intersects_points(
        x_values: np.ndarray,
        y_values: np.ndarray,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
    ) -> bool:
        if x_values.size == 0 or y_values.size == 0:
            return False
        return bool(
            float(np.nanmax(x_values)) >= xmin
            and float(np.nanmin(x_values)) <= xmax
            and float(np.nanmax(y_values)) >= ymin
            and float(np.nanmin(y_values)) <= ymax
        )

    def clear_canvas_selection(self) -> None:
        self.clear_candidate_selection()
        self.selected_roi = None
        self.selected_manual_roi = None
        self.roi_table.clearSelection()
        self.selected_removed_roi = None
        self.invalidate_overlay_cache()
        self.refresh()
        self.update_trace_plot()

    def set_candidate_suite2p_selection(self, indices: set[int], mode: str = "replace", refresh: bool = True) -> None:
        valid = {
            int(idx)
            for idx in indices
            if 0 <= int(idx) < len(self.roi_cache) and self.suite2p_ref_visible(int(idx))
        }
        if mode == "toggle":
            self.selected_candidate_suite2p_refs ^= valid
        elif mode == "add":
            self.selected_candidate_suite2p_refs |= valid
        else:
            self.selected_candidate_suite2p_refs = valid
        if self.selected_candidate_suite2p_refs:
            self.selected_roi = sorted(self.selected_candidate_suite2p_refs)[-1]
            self.selected_manual_roi = None
            self.roi_table.clearSelection()
            if mode != "add":
                self.selected_candidate_cellpose_refs = set()
        else:
            self.selected_roi = None
        if refresh:
            self.invalidate_overlay_cache()
            self.refresh()
            self.update_trace_plot()

    def set_candidate_cellpose_selection(self, indices: set[int], mode: str = "replace") -> None:
        valid = {
            int(idx)
            for idx in indices
            if 0 <= int(idx) < len(self.cellpose_cache) and self.cellpose_ref_visible(int(idx))
        }
        if mode == "toggle":
            self.selected_candidate_cellpose_refs ^= valid
        elif mode == "add":
            self.selected_candidate_cellpose_refs |= valid
        else:
            self.selected_candidate_cellpose_refs = valid
            self.selected_candidate_suite2p_refs = set()
        if self.selected_candidate_cellpose_refs:
            self.selected_roi = None
            self.selected_manual_roi = None
            self.roi_table.clearSelection()
        self.invalidate_overlay_cache()
        self.refresh()
        self.update_trace_plot()

    def select_final_entries(self, entries: list[tuple[str, int]]) -> None:
        wanted = set(entries)
        self.roi_table.blockSignals(True)
        self.roi_table.clearSelection()
        last_entry: tuple[str, int] | None = None
        for row in range(self.roi_table.rowCount()):
            item = self.roi_table.item(row, 0)
            if item is None:
                continue
            kind, idx = item.data(Qt.ItemDataRole.UserRole)
            entry = (str(kind), int(idx))
            if entry in wanted:
                self.roi_table.selectionModel().select(
                    self.roi_table.model().index(row, 0),
                    QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
                )
                last_entry = entry
        self.roi_table.blockSignals(False)
        if last_entry is not None:
            kind, idx = last_entry
            if kind == "suite2p":
                self.selected_roi = idx
                self.selected_manual_roi = None
            else:
                self.selected_roi = None
                self.selected_manual_roi = idx
        else:
            self.selected_roi = None
            self.selected_manual_roi = None
        self.clear_candidate_selection()

    def set_final_entry_selection(self, entry: tuple[str, int], additive: bool = False) -> None:
        current = set(self.selected_final_entries()) if additive else set()
        if additive and entry in current:
            current.remove(entry)
        else:
            current.add(entry)
        self.select_final_entries(sorted(current))
        self.clear_candidate_selection()
        self.invalidate_overlay_cache()
        self.refresh()
        self.update_trace_plot()

    def select_all_final_rois(self) -> None:
        entries = [("suite2p", idx) for idx in sorted(self.selected_suite2p_refs)]
        entries.extend(("manual", idx) for idx in range(len(self.added_rois)))
        self.select_final_entries(entries)
        self.clear_candidate_selection()
        self.invalidate_overlay_cache()
        self.status.showMessage(f"Selected {len(entries)} final ROI(s)")
        self.refresh()
        self.update_trace_plot()

    def selected_removed_indices(self) -> list[int]:
        indices: list[int] = []
        for index in self.removed_roi_table.selectionModel().selectedRows():
            item = self.removed_roi_table.item(index.row(), 0)
            if item is None:
                continue
            indices.append(int(item.data(Qt.ItemDataRole.UserRole)))
        if indices:
            return sorted(set(indices))
        if self.selected_removed_roi is not None:
            return [int(self.selected_removed_roi)]
        return []

    def populate_removed_roi_table(self) -> None:
        self.removed_roi_table.blockSignals(True)
        self.removed_roi_table.setRowCount(0)
        for idx, entry in enumerate(self.removed_rois):
            row = self.removed_roi_table.rowCount()
            self.removed_roi_table.insertRow(row)
            source = str(entry.get("source", ""))
            roi_id = str(entry.get("id", ""))
            roi_type = str(entry.get("type", source))
            for col, value in enumerate([source, roi_id, roi_type]):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter if col == 1 else Qt.AlignmentFlag.AlignLeft)
                if col == 0:
                    item.setData(Qt.ItemDataRole.UserRole, idx)
                self.removed_roi_table.setItem(row, col, item)
            if self.selected_removed_roi == idx:
                self.removed_roi_table.selectRow(row)
        self.removed_roi_table.resizeColumnsToContents()
        self.removed_roi_table.blockSignals(False)

    def rebuild_and_refresh(self) -> None:
        self.invalidate_overlay_cache()
        self.populate_roi_list()
        self.refresh()

    def toggle_suite2p_refs(self) -> None:
        self.show_suite2p_refs = self.show_suite2p.isChecked()
        if self.show_suite2p_refs and not self.suite2p_loaded:
            self.load_suite2p_refs()
        self.invalidate_overlay_cache()
        self.refresh()

    def toggle_cellpose_refs(self) -> None:
        self.show_cellpose_refs = self.show_cellpose.isChecked()
        if self.show_cellpose_refs and not self.cellpose_loaded:
            self.load_cellpose_refs()
        self.invalidate_overlay_cache()
        self.refresh()

    def refresh_trial_list(self, selected_trial_id: str | None = None) -> None:
        self.trial_ids = discover_trial_ids(self.data_root, self.movie_kind)
        self.refresh_trial_selectors(selected_trial_id)

    def trial_date_label(self, trial_id: str) -> str:
        try:
            movie_path = find_movie_path(self.data_root, trial_id, self.movie_kind)
        except Exception:
            return "no date"
        rel_parent = trial_rel_parent(movie_path.parent, step_root_for_movie_kind(self.data_root, self.movie_kind))
        return "no date" if str(rel_parent) in ("", ".") else str(rel_parent)

    def refresh_trial_selectors(self, selected_trial_id: str | None = None) -> None:
        selected_date = self.trial_date_label(selected_trial_id) if selected_trial_id else "__all__"
        dates = sorted({self.trial_date_label(trial_id) for trial_id in self.trial_ids})
        self.date_combo.blockSignals(True)
        self.date_combo.clear()
        self.date_combo.addItem("all dates", "__all__")
        for date_label in dates:
            self.date_combo.addItem(date_label, date_label)
        date_index = self.date_combo.findData(selected_date)
        self.date_combo.setCurrentIndex(date_index if date_index >= 0 else 0)
        self.date_combo.blockSignals(False)
        self.refresh_trial_combo_for_date(selected_trial_id=selected_trial_id)

    def refresh_trial_combo_for_date(self, *_args, selected_trial_id: str | None = None) -> None:
        date_filter = self.date_combo.currentData()
        show_dates = date_filter in (None, "__all__")
        trial_ids = [
            trial_id
            for trial_id in self.trial_ids
            if show_dates or self.trial_date_label(trial_id) == str(date_filter)
        ]
        self.trial_combo.blockSignals(True)
        self.trial_combo.clear()
        for trial_id in trial_ids:
            label = trial_display_label(self.data_root, self.movie_kind, trial_id) if show_dates else trial_id
            self.trial_combo.addItem(label, trial_id)
            if selected_trial_id == trial_id:
                self.trial_combo.setCurrentText(label)
        if not self.trial_combo.currentText() and self.trial_combo.count() > 0:
            self.trial_combo.setCurrentIndex(0)
        self.trial_combo.blockSignals(False)

    def selected_trial_combo_id(self) -> str:
        data = self.trial_combo.currentData()
        if data:
            return str(data)
        text = self.trial_combo.currentText().strip()
        if text in self.trial_ids:
            return text
        if " / " in text:
            candidate = text.rsplit(" / ", 1)[-1].strip()
            if candidate in self.trial_ids:
                return candidate
        for row in range(self.trial_combo.count()):
            if self.trial_combo.itemText(row) == text:
                item_data = self.trial_combo.itemData(row)
                if item_data:
                    return str(item_data)
        return text

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
        save_last_trial_setting(self.data_root, self.movie_kind, self.paths.trial_id)
        self.movie = movie_as_tyx(paths.movie_path)
        self.refresh_acquisition_fps(paths)
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
        self.invalidate_all_image_caches()
        self.recompute_manual_roi_traces()
        self.refresh_trial_list(selected_trial_id=self.paths.trial_id)
        self.refresh()
        self.update_trace_plot()
        self.status.showMessage(f"Switched movie source to {new_kind}; ROI edits were kept unsaved")
        if was_playing:
            self.start_playback()

    def recompute_manual_roi_traces(self) -> None:
        refreshed: list[dict] = []
        saved_rois = self.added_rois
        for idx, saved in enumerate(saved_rois):
            points = [(float(x), float(y)) for x, y in saved.get("points", [])]
            if len(points) < 3:
                continue
            mask = polygon_mask(points, self.movie.shape[-2:])
            if int(mask.sum()) < 3:
                continue
            self.added_rois = saved_rois
            roi = self.build_manual_roi(mask, exclude_manual_idx=idx, compute_traces=True)
            computed_trace_keys = {"f0", "trace_summary", "trace_movie_path", "neuropil_pixels"}
            roi.update(
                {
                    k: v
                    for k, v in saved.items()
                    if not k.startswith("_trace_") and k not in computed_trace_keys
                }
            )
            roi["points"] = points
            refreshed.append(roi)
        self.added_rois = refreshed
        self._manual_traces_dirty = False

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

    def load_cellpose_refs(self) -> None:
        stat_path = cellpose_plane_dir(self.data_root, self.paths.rel_parent, self.paths.trial_id) / "cellpose_stat.npy"
        if not stat_path.exists():
            self.show_cellpose.blockSignals(True)
            self.show_cellpose.setChecked(False)
            self.show_cellpose.blockSignals(False)
            self.show_cellpose_refs = False
            QMessageBox.warning(self, "cellpose refs", "No Cellpose ROI output was found for this trial.")
            return
        stat = np.load(stat_path, allow_pickle=True)
        self.cellpose_cache = self.normalize_candidate_cache(stat)
        self.cellpose_loaded = True
        self.status.showMessage(f"Loaded {len(self.cellpose_cache)} Cellpose reference ROIs")

    def load_saved_curation_if_present(self) -> None:
        out_dir = self.paths.output_dir
        additions_path = out_dir / f"{self.paths.trial_id}_manual_added_rois.json"
        roi_set_path = out_dir / f"{self.paths.trial_id}_manual_roi_set.json"
        selected_path = out_dir / f"{self.paths.trial_id}_selected_suite2p_indices.csv"
        deleted_path = out_dir / f"{self.paths.trial_id}_deleted_suite2p_indices.csv"
        loaded_parts: list[str] = []

        saved_roi_set = self.load_saved_roi_set(roi_set_path)
        saved_suite2p_points = self.saved_suite2p_points_by_id(saved_roi_set)
        saved_suite2p_ids = set(saved_suite2p_points)
        selected_refs = read_index_csv(selected_path)
        deleted_refs = read_index_csv(deleted_path)
        suite2p_refs_to_restore = set(selected_refs) if selected_refs else saved_suite2p_ids
        valid_selected: set[int] = set()
        stale_saved_suite2p_refs: set[int] = set()
        if suite2p_refs_to_restore or deleted_refs:
            if not self.suite2p_loaded:
                self.load_suite2p_refs()
            for idx in suite2p_refs_to_restore:
                saved_points = saved_suite2p_points.get(idx)
                if saved_points is None:
                    if 0 <= idx < len(self.iscell):
                        valid_selected.add(idx)
                    else:
                        stale_saved_suite2p_refs.add(idx)
                    continue
                if 0 <= idx < len(self.iscell):
                    if self.saved_suite2p_shape_matches_current(idx, saved_points):
                        valid_selected.add(idx)
                    else:
                        stale_saved_suite2p_refs.add(idx)
                else:
                    stale_saved_suite2p_refs.add(idx)
            if stale_saved_suite2p_refs:
                stale_saved_suite2p_refs |= valid_selected
                valid_selected = set()
            self.selected_suite2p_refs = valid_selected
            for idx in valid_selected:
                self.iscell[idx, 0] = 1.0
                self.iscell[idx, 1] = 1.0
            if valid_selected:
                self.show_suite2p_refs = True
                self.show_suite2p.blockSignals(True)
                self.show_suite2p.setChecked(True)
                self.show_suite2p.blockSignals(False)
            if valid_selected:
                loaded_parts.append(f"{len(valid_selected)} current suite2p pick(s)")

        saved_rois_for_additions: list[dict] = [
            saved
            for saved in saved_roi_set
            if saved.get("roi_source") not in {"suite2p_reference", "suite2p"}
        ]
        loaded_non_suite2p_from = "manual_roi_set"
        if not saved_rois_for_additions and additions_path.exists():
            try:
                with additions_path.open("r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                    saved_rois_for_additions = loaded if isinstance(loaded, list) else []
                    loaded_non_suite2p_from = "manual_added_rois"
            except Exception as exc:
                QMessageBox.warning(self, "Load saved ROI", f"Could not load saved manual ROIs:\n{exc}")
                saved_rois_for_additions = []

        if saved_rois_for_additions:
            self.added_rois = []
            for saved in saved_rois_for_additions:
                points = [(float(x), float(y)) for x, y in saved.get("points", [])]
                if len(points) < 3:
                    continue
                mask = polygon_mask(points, self.movie.shape[-2:])
                if int(mask.sum()) < 3:
                    continue
                roi = self.build_manual_roi(mask, compute_traces=False)
                roi.update({k: v for k, v in saved.items() if not k.startswith("_trace_")})
                roi["status"] = "accepted"
                roi["points"] = points
                if saved.get("status", "accepted") != "rejected":
                    self.added_rois.append(roi)
            self.mark_manual_traces_dirty()
            loaded_parts.append(f"{len(self.added_rois)} non-suite2p final ROI(s) from {loaded_non_suite2p_from}")

        imported = self.import_saved_suite2p_picks_as_manual(saved_roi_set, stale_saved_suite2p_refs)
        if imported:
            loaded_parts.append(f"{imported} stale suite2p ROI(s) converted to manual")
            self.mark_manual_traces_dirty()

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
        current_area = int(current_mask.sum())
        saved_area = int(saved_mask.sum())
        if current_area == 0 or saved_area == 0:
            return False
        area_ratio = current_area / saved_area
        current_y, current_x = np.nonzero(current_mask)
        saved_y, saved_x = np.nonzero(saved_mask)
        centroid_dist = float(
            np.hypot(
                float(np.mean(current_x) - np.mean(saved_x)),
                float(np.mean(current_y) - np.mean(saved_y)),
            )
        )
        return (intersection / union) >= 0.75 and 0.7 <= area_ratio <= 1.4 and centroid_dist <= 5.0

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
            roi = self.build_manual_roi(mask, compute_traces=False)
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

    def maybe_save_before_switch(self, reason: str = "switching") -> bool:
        if not self.dirty:
            return True
        if reason == "closing":
            message = "Current ROI edits are not saved. Save before closing?"
        else:
            message = "Current ROI edits are not saved. Save before switching?"
        response = QMessageBox.question(
            self,
            "Unsaved ROI edits",
            message,
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

    def save_and_close(self) -> None:
        self.save_outputs(show_message=False)
        self.close()

    def request_close(self) -> None:
        self.close()

    def load_selected_trial(self) -> None:
        trial_id = self.selected_trial_combo_id()
        if not trial_id:
            return
        self.load_trial_id(trial_id)

    def load_next_trial(self) -> None:
        date_filter = self.date_combo.currentData()
        trial_ids = [
            trial_id
            for trial_id in self.trial_ids
            if date_filter in (None, "__all__") or self.trial_date_label(trial_id) == str(date_filter)
        ]
        if not trial_ids:
            return
        try:
            current_idx = trial_ids.index(self.paths.trial_id)
        except ValueError:
            current_idx = -1
        next_idx = (current_idx + 1) % len(trial_ids)
        next_trial_id = trial_ids[next_idx]
        show_dates = date_filter in (None, "__all__")
        label = trial_display_label(self.data_root, self.movie_kind, next_trial_id) if show_dates else next_trial_id
        self.trial_combo.setCurrentText(label)
        self.load_trial_id(next_trial_id)

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
        save_last_trial_setting(self.data_root, self.movie_kind, self.paths.trial_id)
        self.setWindowTitle(f"ROI curation - {paths.trial_id}")
        self.stat = np.array([], dtype=object)
        self.roi_cache = []
        self.iscell = np.zeros((0, 2), dtype=np.float32)
        self.F = None
        self.Fneu = None
        self.suite2p_loaded = False
        self.suite2p_ref_flags = np.zeros((0,), dtype=bool)
        self.cellpose_loaded = False
        self.cellpose_cache = []
        self.movie = movie_as_tyx(paths.movie_path)
        self.load_trace_movie_for_trial(paths)
        self.refresh_acquisition_fps(paths)
        self.frame_index = 0
        self.selected_roi = None
        self.selected_manual_roi = None
        self.clear_candidate_selection()
        self.selection_box_start = None
        self.selection_box_current = None
        self.selection_box_target = None
        self.show_suite2p_refs = False
        self.show_suite2p.blockSignals(True)
        self.show_suite2p.setChecked(False)
        self.show_suite2p.blockSignals(False)
        self.show_cellpose_refs = False
        self.show_cellpose.blockSignals(True)
        self.show_cellpose.setChecked(False)
        self.show_cellpose.blockSignals(False)
        self.selected_suite2p_refs = set()
        self.added_rois = []
        self.removed_rois = []
        self.selected_removed_roi = None
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
        self.invalidate_all_image_caches()
        self.left_canvas.reset_view(emit=False)
        self.right_canvas.reset_view(emit=False)
        self.refresh()
        self.update_trace_plot()
        if was_playing:
            self.start_playback()

    def refresh_acquisition_fps(self, paths: TrialPaths) -> None:
        metadata_path = find_metadata_path(self.data_root, paths.trial_id, paths.rel_parent)
        self.acquisition_fps, self.acquisition_fps_source = fps_from_metadata_path(metadata_path, default_fps=2.0)
        if hasattr(self, "playback_info_label"):
            self.update_play_timer_interval()

    def effective_playback_fps(self) -> float:
        if self.playback_mode_combo.currentData() == "real-time":
            return max(float(self.acquisition_fps) * float(self.speed_spin.value()), 0.1)
        return max(float(self.fps_spin.value()), 0.1)

    def update_playback_controls(self) -> None:
        real_time = self.playback_mode_combo.currentData() == "real-time"
        self.speed_spin.setVisible(real_time)
        self.fps_spin.setVisible(not real_time)
        self.update_play_timer_interval()

    def update_play_timer_interval(self) -> None:
        fps = self.effective_playback_fps()
        self.play_timer.setInterval(max(1, int(round(1000.0 / fps))))
        mode = "real" if self.playback_mode_combo.currentData() == "real-time" else "display"
        self.playback_info_label.setText(
            f"{fps:.2g} fps ({mode}; acquired {self.acquisition_fps:.3g} fps)"
        )

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
        self.invalidate_frame_cache()
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
        self.invalidate_frame_cache()
        self.refresh()

    def toggle_playback(self) -> None:
        if self.playing:
            self.pause_playback()
        else:
            self.start_playback()

    def start_playback(self) -> None:
        self.playing = True
        self.play_btn.setText("Pause")
        self.invalidate_overlay_cache()
        self.update_play_timer_interval()
        self.play_timer.start()

    def pause_playback(self) -> None:
        self.playing = False
        self.play_btn.setText("Play")
        self.play_timer.stop()
        self.invalidate_overlay_cache()
        self.refresh()

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

    def select_roi_from_table(self, row: int, _column: int, _previous_row: int, _previous_column: int) -> None:
        item = self.roi_table.item(row, 0)
        if item is None:
            return
        self.selected_removed_roi = None
        self.removed_roi_table.clearSelection()
        self.selected_candidate_suite2p_refs = set()
        self.selected_candidate_cellpose_refs = set()
        kind, idx = item.data(Qt.ItemDataRole.UserRole)
        if kind == "suite2p":
            self.selected_roi = int(idx)
            self.selected_manual_roi = None
        else:
            self.selected_roi = None
            self.selected_manual_roi = int(idx)
        self.refresh()
        self.update_trace_plot()

    def select_removed_roi_from_table(self, row: int, _column: int, _previous_row: int, _previous_column: int) -> None:
        item = self.removed_roi_table.item(row, 0)
        if item is None:
            return
        idx = item.data(Qt.ItemDataRole.UserRole)
        self.selected_removed_roi = int(idx)
        self.selected_roi = None
        self.selected_manual_roi = None
        self.clear_candidate_selection()
        self.roi_table.clearSelection()
        self.status.showMessage("Selected removed ROI; Restore removed will put it back in the final ROI table")

    def handle_overlay_click(self, x: float, y: float, button: int) -> None:
        if (
            button == Qt.MouseButton.LeftButton
            and (self.show_suite2p_refs or self.show_cellpose_refs)
            and QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self.selection_box_start = (float(x), float(y))
            self.selection_box_current = (float(x), float(y))
            self.selection_box_target = "candidate_refs"
            self.status.showMessage("Drag to box-select candidate ROIs")
            self.refresh()
            return
        manual_idx = self.manual_roi_at(x, y)
        if manual_idx is not None:
            if button == Qt.MouseButton.RightButton:
                manual_id = self.added_rois[manual_idx].get("manual_roi_id")
                self.delete_manual_roi_at(manual_idx)
                self.status.showMessage(f"Removed manual ROI {manual_id} from final ROI set")
            else:
                self.set_final_entry_selection(
                    ("manual", manual_idx),
                    additive=self.additive_selection_requested(),
                )
                self.status.showMessage(f"Selected manual ROI {self.added_rois[manual_idx]['manual_roi_id']}")
            return
        if not self.show_suite2p_refs and not self.show_cellpose_refs:
            if button == Qt.MouseButton.LeftButton:
                self.clear_canvas_selection()
            return
        best_source, best_idx, best_dist = self.nearest_candidate_roi(x, y)
        if button == Qt.MouseButton.RightButton:
            if best_source == "suite2p" and best_idx is not None and best_dist <= 100:
                if self.remove_picked_suite2p_roi(best_idx):
                    self.status.showMessage(f"Removed suite2p ROI {best_idx} from picked refs")
            return
        if best_idx is not None and best_dist <= 100:
            if button == Qt.MouseButton.LeftButton:
                mode = "toggle" if self.additive_selection_requested() else "replace"
                if best_source == "suite2p":
                    self.set_candidate_suite2p_selection({best_idx}, mode=mode)
                    count = len(self.valid_candidate_suite2p_selection())
                    self.status.showMessage(f"Selected {count} suite2p candidate ROI(s)")
                elif best_source == "cellpose":
                    self.set_candidate_cellpose_selection({best_idx}, mode=mode)
                    count = len(self.valid_candidate_cellpose_selection())
                    self.status.showMessage(f"Selected {count} cellpose candidate ROI(s)")
            return
        if button == Qt.MouseButton.LeftButton:
            self.clear_canvas_selection()
            self.status.showMessage("Cleared ROI selection")


    def handle_overlay_double_click(self, x: float, y: float, button: object) -> None:
        if (not self.show_suite2p_refs and not self.show_cellpose_refs) or button != Qt.MouseButton.LeftButton:
            return
        best_source, best_idx, best_dist = self.nearest_candidate_roi(x, y)
        if best_idx is None or best_dist > 100:
            return
        if best_source == "suite2p":
            self.selected_roi = best_idx
            self.selected_manual_roi = None
            self.selected_candidate_suite2p_refs = {best_idx} if best_idx not in self.selected_suite2p_refs else set()
            self.set_selected_state(1 if best_idx not in self.selected_suite2p_refs else 0)
            self.status.showMessage(f"Toggled suite2p ROI {best_idx}")
        elif best_source == "cellpose" and self.add_cellpose_candidate_to_final(best_idx):
            self.status.showMessage(f"Added cellpose ROI {best_idx} to final set")

    def suite2p_ref_entries_in_box(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> set[int]:
        x0, y0 = start
        x1, y1 = end
        xmin, xmax = sorted((float(x0), float(x1)))
        ymin, ymax = sorted((float(y0), float(y1)))
        indices: set[int] = set()
        for idx, roi in enumerate(self.roi_cache):
            if not self.suite2p_ref_visible(idx):
                continue
            xpix = np.asarray(roi.get("xpix", []), dtype=float)
            ypix = np.asarray(roi.get("ypix", []), dtype=float)
            if self.rect_intersects_points(xpix, ypix, xmin, xmax, ymin, ymax):
                indices.add(int(idx))
        return indices

    def handle_overlay_drag(self, x: float, y: float, buttons: object) -> None:
        if (
            self.selection_box_target == "candidate_refs"
            and self.selection_box_start is not None
            and (buttons & Qt.MouseButton.LeftButton)
        ):
            self.selection_box_current = (float(x), float(y))
            self.refresh()

    def handle_overlay_release(self, x: float, y: float, button: object) -> None:
        if (
            self.selection_box_target == "candidate_refs"
            and button == Qt.MouseButton.LeftButton
            and self.selection_box_start is not None
        ):
            self.selection_box_current = (float(x), float(y))
            suite2p_indices, cellpose_indices = self.candidate_entries_in_box(self.selection_box_start, self.selection_box_current)
            mode = "add" if self.additive_selection_requested() else "replace"
            self.selection_box_start = None
            self.selection_box_current = None
            self.selection_box_target = None
            self.set_candidate_suite2p_selection(suite2p_indices, mode=mode, refresh=False)
            self.set_candidate_cellpose_selection(cellpose_indices, mode="add" if mode == "add" or suite2p_indices else "replace")
            total = len(self.valid_candidate_suite2p_selection()) + len(self.valid_candidate_cellpose_selection())
            self.status.showMessage(f"Selected {total} candidate ROI(s)")

    def nearest_existing_roi(self, x: float, y: float) -> tuple[int | None, float]:
        best_idx = None
        best_dist = float("inf")
        for idx, roi in enumerate(self.roi_cache):
            if not self.suite2p_ref_visible(idx):
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

    def nearest_candidate_roi(self, x: float, y: float) -> tuple[str | None, int | None, float]:
        best_source: str | None = None
        best_idx: int | None = None
        best_dist = float("inf")
        if self.show_suite2p_refs:
            idx, dist = self.nearest_existing_roi(x, y)
            if idx is not None and dist < best_dist:
                best_source, best_idx, best_dist = "suite2p", idx, dist
        if self.show_cellpose_refs:
            final_cellpose_ids = self.cellpose_final_original_ids()
            for idx, roi in enumerate(self.cellpose_cache):
                if idx in final_cellpose_ids:
                    continue
                ypix = np.asarray(roi.get("ypix", []), dtype=float)
                xpix = np.asarray(roi.get("xpix", []), dtype=float)
                if ypix.size == 0:
                    continue
                dist = float(np.min((xpix - x) ** 2 + (ypix - y) ** 2))
                if dist < best_dist:
                    best_source, best_idx, best_dist = "cellpose", idx, dist
        return best_source, best_idx, best_dist

    def candidate_entries_in_box(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> tuple[set[int], set[int]]:
        suite2p_indices = self.suite2p_ref_entries_in_box(start, end) if self.show_suite2p_refs else set()
        x0, y0 = start
        x1, y1 = end
        xmin, xmax = sorted((float(x0), float(x1)))
        ymin, ymax = sorted((float(y0), float(y1)))
        cellpose_indices: set[int] = set()
        if self.show_cellpose_refs:
            final_cellpose_ids = self.cellpose_final_original_ids()
            for idx, roi in enumerate(self.cellpose_cache):
                if idx in final_cellpose_ids:
                    continue
                xpix = np.asarray(roi.get("xpix", []), dtype=float)
                ypix = np.asarray(roi.get("ypix", []), dtype=float)
                if self.rect_intersects_points(xpix, ypix, xmin, xmax, ymin, ymax):
                    cellpose_indices.add(int(idx))
        return suite2p_indices, cellpose_indices

    def nearest_selected_suite2p_roi(self, x: float, y: float) -> tuple[int | None, float]:
        best_idx = None
        best_dist = float("inf")
        for idx in sorted(self.selected_suite2p_refs):
            if idx >= len(self.roi_cache):
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
            points = roi.get("points", [])
            if len(points) >= 3 and MplPath(points).contains_point((x, y)):
                return idx
        return None

    def final_roi_entries_in_box(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> list[tuple[str, int]]:
        x0, y0 = start
        x1, y1 = end
        xmin, xmax = sorted((float(x0), float(x1)))
        ymin, ymax = sorted((float(y0), float(y1)))
        entries: list[tuple[str, int]] = []
        for idx in sorted(self.selected_suite2p_refs):
            if idx >= len(self.roi_cache):
                continue
            roi = self.roi_cache[idx]
            xpix = np.asarray(roi.get("xpix", []), dtype=float)
            ypix = np.asarray(roi.get("ypix", []), dtype=float)
            if self.rect_intersects_points(xpix, ypix, xmin, xmax, ymin, ymax):
                entries.append(("suite2p", idx))
        for idx, roi in enumerate(self.added_rois):
            points = np.asarray(roi.get("points", []), dtype=float)
            if points.ndim == 2 and points.shape[1] >= 2 and self.rect_intersects_points(
                points[:, 0],
                points[:, 1],
                xmin,
                xmax,
                ymin,
                ymax,
            ):
                entries.append(("manual", idx))
        return entries

    def handle_right_click(self, x: float, y: float, button: int) -> None:
        mode = self.mode_combo.currentText()
        final_rois_visible = self.show_final_rois.isChecked()
        if (
            mode == "select ROI"
            and button == Qt.MouseButton.LeftButton
            and final_rois_visible
            and QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self.selection_box_start = (float(x), float(y))
            self.selection_box_current = (float(x), float(y))
            self.selection_box_target = "final"
            self.status.showMessage("Drag to box-select final ROIs")
            self.refresh()
            return
        if button == Qt.MouseButton.RightButton and final_rois_visible:
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
            if not final_rois_visible:
                return
            manual_idx = self.manual_roi_at(x, y)
            if manual_idx is not None:
                if button == Qt.MouseButton.RightButton:
                    manual_id = self.added_rois[manual_idx].get("manual_roi_id")
                    self.delete_manual_roi_at(manual_idx)
                    self.status.showMessage(f"Removed manual ROI {manual_id} from final ROI set")
                else:
                    self.set_final_entry_selection(
                        ("manual", manual_idx),
                        additive=self.additive_selection_requested(),
                    )
                return
            suite_idx, suite_dist = self.nearest_selected_suite2p_roi(x, y)
            if suite_idx is not None and suite_dist <= 100:
                if button == Qt.MouseButton.RightButton:
                    if self.remove_picked_suite2p_roi(suite_idx):
                        self.status.showMessage(f"Removed suite2p ROI {suite_idx} from picked refs")
                else:
                    self.set_final_entry_selection(
                        ("suite2p", suite_idx),
                        additive=self.additive_selection_requested(),
                    )
                return
            if button == Qt.MouseButton.LeftButton:
                self.clear_canvas_selection()
                self.status.showMessage("Cleared ROI selection")
            return

    def handle_right_drag(self, x: float, y: float, buttons: object) -> None:
        mode = self.mode_combo.currentText()
        if (
            mode == "select ROI"
            and self.selection_box_target == "final"
            and self.selection_box_start is not None
            and (buttons & Qt.MouseButton.LeftButton)
        ):
            self.selection_box_current = (float(x), float(y))
            self.refresh()
            return
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
        if (
            mode == "select ROI"
            and self.selection_box_target == "final"
            and button == Qt.MouseButton.LeftButton
            and self.selection_box_start is not None
        ):
            self.selection_box_current = (float(x), float(y))
            entries = self.final_roi_entries_in_box(self.selection_box_start, self.selection_box_current)
            if self.additive_selection_requested():
                current = set(self.selected_final_entries())
                current.update(entries)
                entries = sorted(current)
            self.selection_box_start = None
            self.selection_box_current = None
            self.selection_box_target = None
            self.select_final_entries(entries)
            self.status.showMessage(f"Selected {len(entries)} final ROI(s)")
            self.refresh()
            self.update_trace_plot()
            return
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
        roi = self.build_manual_roi(mask, compute_traces=False)
        self.added_rois.append(roi)
        self.selected_roi = None
        self.selected_manual_roi = len(self.added_rois) - 1
        self.current_polygon = []
        self.mark_manual_traces_dirty()
        self.invalidate_overlay_cache()
        self.dirty = True
        self.status.showMessage(f"Added {roi['roi_type']} ROI {roi['manual_roi_id']} with {roi['n_pixels']} pixels")
        self.populate_roi_list()
        self.refresh()
        self.update_trace_plot()

    def final_roi_exclusion_mask(
        self,
        shape: tuple[int, int],
        exclude_manual_idx: int | None = None,
        exclude_suite2p_idx: int | None = None,
    ) -> np.ndarray:
        exclusion = np.zeros(shape, dtype=bool)
        for idx in sorted(self.selected_suite2p_refs):
            if idx == exclude_suite2p_idx or idx >= len(self.roi_cache):
                continue
            roi = self.roi_cache[idx]
            ypix = np.asarray(roi.get("ypix", []), dtype=int)
            xpix = np.asarray(roi.get("xpix", []), dtype=int)
            valid = (ypix >= 0) & (ypix < shape[0]) & (xpix >= 0) & (xpix < shape[1])
            exclusion[ypix[valid], xpix[valid]] = True
        for idx, roi in enumerate(self.added_rois):
            if idx == exclude_manual_idx or roi.get("status", "accepted") != "accepted":
                continue
            points = [(float(x), float(y)) for x, y in roi.get("points", [])]
            if len(points) < 3:
                continue
            exclusion |= polygon_mask(points, shape)
        return exclusion

    def build_manual_roi(
        self,
        mask: np.ndarray,
        exclude_manual_idx: int | None = None,
        points: list[tuple[float, float]] | None = None,
        roi_type: str | None = None,
        extra: dict | None = None,
        compute_traces: bool = True,
    ) -> dict:
        ypix, xpix = np.nonzero(mask)
        roi = {
            "manual_roi_id": self.next_manual_roi_id(),
            "roi_type": roi_type or ("ellipse" if self.mode_combo.currentText() == "draw ellipse ROI" else "freehand"),
            "status": "accepted",
            "points": [[float(x), float(y)] for x, y in (points if points is not None else self.current_polygon)],
            "frame_added": int(self.frame_index),
            "n_pixels": int(mask.sum()),
            "x_mean": float(np.mean(xpix)),
            "y_mean": float(np.mean(ypix)),
            "neuropil_coeff": float(self.neuropil_coeff),
            "f0_percentile": float(self.f0_percentile),
            "f0": None,
            "trace_movie_path": str(self.trace_movie_path),
            "neuropil_exclusion": "other_final_rois",
            "neuropil_pixels": 0,
        }
        if compute_traces:
            f, fneu, fcorr, dff, f0, neuropil_pixels = self.traces_for_mask(
                mask,
                exclusion_mask=self.final_roi_exclusion_mask(mask.shape, exclude_manual_idx=exclude_manual_idx),
            )
            roi.update(
                {
                    "f0": float(f0),
                    "neuropil_pixels": int(neuropil_pixels),
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
            )
        if extra:
            roi.update(extra)
        return roi

    def traces_for_mask(
        self,
        mask: np.ndarray,
        exclusion_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
        annulus = annulus_mask(mask, exclusion_mask=exclusion_mask)
        f = self.mean_trace(mask)
        fneu = self.mean_trace(annulus) if np.any(annulus) else np.full_like(f, np.nan)
        if np.all(~np.isfinite(fneu)):
            fcorr = f.copy()
        else:
            fcorr = f - self.neuropil_coeff * np.nan_to_num(fneu, nan=0.0)
        dff, f0 = robust_dff(fcorr, self.f0_percentile, self.f0_eps)
        return f, fneu, fcorr, dff, f0, int(np.count_nonzero(annulus))

    def suite2p_mask(self, roi_idx: int) -> np.ndarray:
        mask = np.zeros(self.trace_movie.shape[-2:], dtype=bool)
        if roi_idx < 0 or roi_idx >= len(self.roi_cache):
            return mask
        roi = self.roi_cache[roi_idx]
        ypix = np.asarray(roi.get("ypix", []), dtype=int)
        xpix = np.asarray(roi.get("xpix", []), dtype=int)
        valid = (
            (ypix >= 0)
            & (ypix < mask.shape[0])
            & (xpix >= 0)
            & (xpix < mask.shape[1])
        )
        mask[ypix[valid], xpix[valid]] = True
        return mask

    def cellpose_mask(self, roi_idx: int) -> np.ndarray:
        mask = np.zeros(self.trace_movie.shape[-2:], dtype=bool)
        if roi_idx < 0 or roi_idx >= len(self.cellpose_cache):
            return mask
        roi = self.cellpose_cache[roi_idx]
        ypix = np.asarray(roi.get("ypix", []), dtype=int)
        xpix = np.asarray(roi.get("xpix", []), dtype=int)
        valid = (
            (ypix >= 0)
            & (ypix < mask.shape[0])
            & (xpix >= 0)
            & (xpix < mask.shape[1])
        )
        mask[ypix[valid], xpix[valid]] = True
        return mask

    def cellpose_points(self, roi_idx: int) -> list[tuple[float, float]]:
        if roi_idx < 0 or roi_idx >= len(self.cellpose_cache):
            return []
        roi = self.cellpose_cache[roi_idx]
        ypix = np.asarray(roi.get("boundary_ypix", roi.get("ypix", [])), dtype=np.int32)
        xpix = np.asarray(roi.get("boundary_xpix", roi.get("xpix", [])), dtype=np.int32)
        return ordered_boundary_points(ypix, xpix)

    def add_cellpose_candidate_to_final(
        self,
        roi_idx: int,
        push_undo: bool = True,
        refresh: bool = True,
        compute_traces: bool = False,
    ) -> bool:
        mask = self.cellpose_mask(roi_idx)
        if not np.any(mask):
            return False
        if push_undo:
            self.push_undo("cellpose-pick")
        points = self.cellpose_points(roi_idx)
        roi = self.build_manual_roi(
            mask,
            points=points,
            roi_type="cellpose",
            extra={
                "roi_source": "cellpose",
                "cellpose_original_id": int(roi_idx),
            },
            compute_traces=compute_traces,
        )
        self.added_rois.append(roi)
        self.selected_manual_roi = len(self.added_rois) - 1
        self.selected_roi = None
        self.selected_candidate_cellpose_refs.discard(int(roi_idx))
        self.selected_removed_roi = None
        self.mark_manual_traces_dirty()
        self.invalidate_overlay_cache()
        self.dirty = True
        if refresh:
            self.populate_roi_list()
            self.refresh()
            self.update_trace_plot()
        return True

    def mean_trace(self, mask: np.ndarray) -> np.ndarray:
        trace_movie = self.trace_movie
        if trace_movie.shape[-2:] != mask.shape:
            trace_movie = self.movie
        if not np.any(mask):
            return np.full((trace_movie.shape[0],), np.nan, dtype=np.float32)
        pixels = trace_movie[:, mask]
        return np.asarray(np.nanmean(pixels, axis=1), dtype=np.float32)

    def keep_all_visible_refs(self) -> None:
        suite2p_indices = [
            idx
            for idx in range(len(self.roi_cache))
            if self.show_suite2p_refs and self.suite2p_ref_visible(idx)
        ]
        final_cellpose_ids = self.cellpose_final_original_ids()
        cellpose_indices = [
            idx
            for idx in range(len(self.cellpose_cache))
            if self.show_cellpose_refs and idx not in final_cellpose_ids
        ]
        if not suite2p_indices and not cellpose_indices:
            self.status.showMessage("No visible reference ROIs to keep")
            return
        self.push_undo("keep-all-visible-refs")
        for roi_idx in suite2p_indices:
            self.selected_suite2p_refs.add(int(roi_idx))
            if 0 <= roi_idx < len(self.iscell):
                self.iscell[roi_idx, 0] = 1.0
                self.iscell[roi_idx, 1] = 1.0
        added_cellpose = 0
        for roi_idx in cellpose_indices:
            if self.add_cellpose_candidate_to_final(
                int(roi_idx),
                push_undo=False,
                refresh=False,
                compute_traces=False,
            ):
                added_cellpose += 1
        self.clear_candidate_selection()
        self.selected_roi = suite2p_indices[-1] if suite2p_indices else None
        if added_cellpose:
            self.selected_roi = None
            self.selected_manual_roi = len(self.added_rois) - 1
        self.selected_removed_roi = None
        self.mark_manual_traces_dirty()
        self.invalidate_overlay_cache()
        self.dirty = True
        self.populate_roi_list()
        self.status.showMessage(
            f"Kept {len(suite2p_indices)} suite2p and {added_cellpose} Cellpose visible reference ROI(s)"
        )
        self.refresh()
        self.update_trace_plot()

    def set_selected_state(self, state: int) -> None:
        cellpose_indices = sorted(self.valid_candidate_cellpose_selection())
        if state and cellpose_indices:
            self.push_undo("cellpose-pick-batch")
            added = 0
            for roi_idx in cellpose_indices:
                if self.add_cellpose_candidate_to_final(
                    int(roi_idx),
                    push_undo=False,
                    refresh=False,
                    compute_traces=False,
                ):
                    added += 1
            self.clear_candidate_selection()
            self.mark_manual_traces_dirty()
            self.invalidate_overlay_cache()
            self.dirty = True
            self.populate_roi_list()
            self.status.showMessage(f"Added {added} cellpose candidate ROI(s) to final set")
            self.refresh()
            self.update_trace_plot()
            return

        candidate_indices = sorted(self.valid_candidate_suite2p_selection())
        if state and candidate_indices:
            self.push_undo("suite2p-pick-batch")
            for roi_idx in candidate_indices:
                self.selected_suite2p_refs.add(int(roi_idx))
                if 0 <= roi_idx < len(self.iscell):
                    self.iscell[roi_idx, 0] = 1.0
                    self.iscell[roi_idx, 1] = 1.0
            self.clear_candidate_selection()
            self.selected_roi = candidate_indices[-1]
            self.selected_manual_roi = None
            self.mark_manual_traces_dirty()
            self.invalidate_overlay_cache()
            self.dirty = True
            self.populate_roi_list()
            self.status.showMessage(f"Added {len(candidate_indices)} suite2p candidate ROI(s) to final set")
            self.refresh()
            self.update_trace_plot()
            return
        if not state and len(self.selected_final_entries()) > 1:
            self.delete_selected()
            return
        if self.selected_manual_roi is not None and 0 <= self.selected_manual_roi < len(self.added_rois):
            if state:
                self.status.showMessage("Manual ROI is already in the final ROI set")
            else:
                self.delete_manual_roi_at(self.selected_manual_roi)
            return
        if self.selected_roi is None:
            return
        self.push_undo("suite2p-pick")
        roi_idx = int(self.selected_roi)
        was_selected = roi_idx in self.selected_suite2p_refs
        if state:
            self.selected_suite2p_refs.add(roi_idx)
            self.selected_candidate_suite2p_refs.discard(roi_idx)
            self.selected_removed_roi = None
            self.iscell[roi_idx, 0] = 1.0
            self.iscell[roi_idx, 1] = 1.0
        else:
            self.selected_suite2p_refs.discard(roi_idx)
            if was_selected:
                self.selected_candidate_suite2p_refs.add(roi_idx)
                self.selected_removed_roi = None
            self.iscell[roi_idx, 0] = 0.0
            self.iscell[roi_idx, 1] = 0.0
        self.mark_manual_traces_dirty()
        self.invalidate_overlay_cache()
        self.dirty = True
        self.populate_roi_list()
        self.refresh()
        self.update_trace_plot()

    def remove_picked_suite2p_roi(self, roi_idx: int, push_undo: bool = True) -> bool:
        if roi_idx not in self.selected_suite2p_refs:
            return False
        if push_undo:
            self.push_undo("suite2p-unpick")
        self.selected_suite2p_refs.discard(int(roi_idx))
        self.selected_candidate_suite2p_refs.add(int(roi_idx))
        self.selected_removed_roi = None
        if 0 <= roi_idx < len(self.iscell):
            self.iscell[roi_idx, 0] = 0.0
            self.iscell[roi_idx, 1] = 0.0
        if self.selected_roi == roi_idx:
            self.selected_roi = None
        self.mark_manual_traces_dirty()
        self.invalidate_overlay_cache()
        self.dirty = True
        self.populate_roi_list()
        self.refresh()
        self.update_trace_plot()
        return True

    def delete_manual_roi_at(self, manual_idx: int, push_undo: bool = True) -> None:
        if manual_idx < 0 or manual_idx >= len(self.added_rois):
            return
        if push_undo:
            self.push_undo("delete-manual")
        removed = self.added_rois.pop(manual_idx)
        self.removed_rois.append(
            {
                "source": str(removed.get("roi_source", "manual")),
                "id": int(removed.get("cellpose_original_id", removed.get("manual_roi_id", manual_idx))),
                "type": str(removed.get("roi_type", "manual")),
                "roi": copy.deepcopy(removed),
            }
        )
        self.selected_removed_roi = None
        if self.selected_manual_roi == manual_idx:
            self.selected_manual_roi = None
        elif self.selected_manual_roi is not None and self.selected_manual_roi > manual_idx:
            self.selected_manual_roi -= 1
        self.mark_manual_traces_dirty()
        self.invalidate_overlay_cache()
        self.dirty = True
        self.status.showMessage(f"Removed manual ROI {removed.get('manual_roi_id')} from final ROI set")
        self.populate_roi_list()
        self.refresh()
        self.update_trace_plot()

    def restore_removed_roi(self) -> None:
        indices = self.selected_removed_indices()
        if not indices:
            self.status.showMessage("Select a removed ROI to restore")
            return
        self.push_undo("restore-removed")
        self.selected_removed_roi = None
        restored_count = 0
        for idx in sorted(indices, reverse=True):
            if idx < 0 or idx >= len(self.removed_rois):
                continue
            restored = self.removed_rois.pop(idx)
            source = restored.get("source")
            if source == "suite2p":
                roi_idx = int(restored["id"])
                self.selected_suite2p_refs.add(roi_idx)
                if 0 <= roi_idx < len(self.iscell):
                    self.iscell[roi_idx, 0] = 1.0
                    self.iscell[roi_idx, 1] = 1.0
                self.selected_roi = roi_idx
                self.selected_manual_roi = None
                restored_count += 1
            elif source in {"manual", "cellpose", "suite2p_reference_imported"} and "roi" in restored:
                self.added_rois.append(copy.deepcopy(restored["roi"]))
                self.selected_roi = None
                self.selected_manual_roi = len(self.added_rois) - 1
                restored_count += 1
        self.status.showMessage(f"Restored {restored_count} ROI(s)")
        self.mark_manual_traces_dirty()
        self.invalidate_overlay_cache()
        self.dirty = True
        self.populate_roi_list()
        self.refresh()
        self.update_trace_plot()

    def delete_selected(self) -> None:
        entries = self.selected_final_entries()
        if not entries:
            return
        self.push_undo("remove-selected")
        removed_count = 0
        suite2p_indices = [idx for kind, idx in entries if kind == "suite2p"]
        manual_indices = [idx for kind, idx in entries if kind == "manual"]
        for roi_idx in suite2p_indices:
            if self.remove_picked_suite2p_roi(roi_idx, push_undo=False):
                removed_count += 1
        for manual_idx in sorted(set(manual_indices), reverse=True):
            before = len(self.added_rois)
            self.delete_manual_roi_at(manual_idx, push_undo=False)
            if len(self.added_rois) < before:
                removed_count += 1
        self.status.showMessage(f"Removed {removed_count} ROI(s) from final ROI set")

    def clear_final_rois(self) -> None:
        if not self.selected_suite2p_refs and not self.added_rois:
            self.status.showMessage("No final ROIs to clear")
            return
        self.push_undo("clear-final-rois")
        cleared_count = len(self.selected_suite2p_refs) + len(self.added_rois)
        for idx in sorted(self.selected_suite2p_refs):
            if 0 <= idx < len(self.iscell):
                self.iscell[idx, 0] = 0.0
                self.iscell[idx, 1] = 0.0
        for roi in self.added_rois:
            self.removed_rois.append(
                {
                    "source": str(roi.get("roi_source", "manual")),
                    "id": int(roi.get("cellpose_original_id", roi.get("manual_roi_id", -1))),
                    "type": str(roi.get("roi_type", "manual")),
                    "roi": copy.deepcopy(roi),
                }
            )
        self.selected_suite2p_refs = set()
        self.added_rois = []
        self.clear_candidate_selection()
        self.selected_removed_roi = None
        self.selected_roi = None
        self.selected_manual_roi = None
        self.mark_manual_traces_dirty()
        self.invalidate_overlay_cache()
        self.dirty = True
        self.populate_roi_list()
        self.status.showMessage(f"Cleared {cleared_count} final ROI(s)")
        self.refresh()
        self.update_trace_plot()

    def push_undo(self, label: str) -> None:
        self.undo_stack.append(
            {
                "label": label,
                "iscell": self.iscell.copy(),
                "selected_candidate_suite2p_refs": set(self.selected_candidate_suite2p_refs),
                "selected_candidate_cellpose_refs": set(self.selected_candidate_cellpose_refs),
                "selected_suite2p_refs": set(self.selected_suite2p_refs),
                "added_rois": copy.deepcopy(self.added_rois),
                "removed_rois": copy.deepcopy(self.removed_rois),
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
        self.selected_candidate_suite2p_refs = state.get("selected_candidate_suite2p_refs", set())
        self.selected_candidate_cellpose_refs = state.get("selected_candidate_cellpose_refs", set())
        self.selected_suite2p_refs = state["selected_suite2p_refs"]
        self.added_rois = state["added_rois"]
        self.removed_rois = state["removed_rois"]
        self.selected_removed_roi = None
        if state["label"] == "add-manual":
            self.current_polygon = []
            self.ellipse_start = None
            self.ellipse_current = None
            self.freehand_drawing = False
        else:
            self.current_polygon = state["current_polygon"]
            self.ellipse_start = state["ellipse_start"]
            self.ellipse_current = state["ellipse_current"]
        self.selected_roi = state["selected_roi"]
        self.selected_manual_roi = state["selected_manual_roi"]
        self.recompute_manual_roi_traces()
        self.invalidate_overlay_cache()
        self.dirty = True
        self.status.showMessage(f"Undid {state['label']}")
        self.populate_roi_list()
        self.refresh()
        self.update_trace_plot()

    def update_trace_plot(self, force: bool = False) -> None:
        if hasattr(self, "auto_trace_check") and not force and not self.auto_trace_check.isChecked():
            if not self._trace_pause_notice_shown:
                self.trace_canvas.plot_empty("Trace paused; press Update trace")
                self._trace_pause_notice_shown = True
            return
        self._trace_pause_notice_shown = False
        if self.selected_manual_roi is not None and 0 <= self.selected_manual_roi < len(self.added_rois):
            self.ensure_manual_traces_current()
            if self.selected_manual_roi is None or self.selected_manual_roi >= len(self.added_rois):
                self.trace_canvas.plot_empty("Select a ROI or draw a freehand/ellipse ROI")
                return
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
            cellpose_indices = sorted(self.valid_candidate_cellpose_selection())
            if cellpose_indices:
                roi_idx = int(cellpose_indices[-1])
                mask = self.cellpose_mask(roi_idx)
                if not np.any(mask):
                    self.trace_canvas.plot_empty("Selected Cellpose ROI has no pixels")
                    return
                f, fneu, fcorr, dff, _, _ = self.traces_for_mask(
                    mask,
                    exclusion_mask=self.final_roi_exclusion_mask(mask.shape),
                )
                self.trace_canvas.plot_traces(f"cellpose ROI {roi_idx}", f, fneu, fcorr, dff)
                return
            self.trace_canvas.plot_empty("Select a ROI or draw a freehand/ellipse ROI")
            return
        mask = self.suite2p_mask(self.selected_roi)
        if not np.any(mask):
            self.trace_canvas.plot_empty("Selected suite2p ROI has no pixels")
            return
        f, fneu, fcorr, dff, _, _ = self.traces_for_mask(
            mask,
            exclusion_mask=self.final_roi_exclusion_mask(mask.shape, exclude_suite2p_idx=self.selected_roi),
        )
        self.trace_canvas.plot_traces(f"suite2p ROI {self.selected_roi}", f, fneu, fcorr, dff)

    def base_frame_pixmap(self) -> QPixmap:
        scale = self.render_scale()
        key = (int(self.frame_index), round(float(self.low_pct), 3), round(float(self.high_pct), 3), round(scale, 3))
        cached = self._base_pixmap_cache.get(key)
        if cached is not None:
            return QPixmap(cached)
        frame = normalize_frame(self.movie[self.frame_index], self.low_pct, self.high_pct)
        if scale < 0.999:
            height, width = frame.shape
            out_h = max(1, int(round(height * scale)))
            out_w = max(1, int(round(width * scale)))
            y_idx = np.linspace(0, height - 1, out_h).astype(np.int32)
            x_idx = np.linspace(0, width - 1, out_w).astype(np.int32)
            frame = frame[np.ix_(y_idx, x_idx)]
        frame = np.ascontiguousarray(frame)
        height, width = frame.shape
        qimg = QImage(frame.data, width, height, width, QImage.Format.Format_Grayscale8).copy()
        pixmap = QPixmap.fromImage(qimg)
        self._base_pixmap_cache.clear()
        self._base_pixmap_cache[key] = QPixmap(pixmap)
        return pixmap

    def overlay_cache_key(self, view: str) -> tuple:
        if view == "reference":
            return (
                view,
                self._roi_overlay_revision,
                round(self.render_scale(), 3),
                bool(self.show_suite2p_refs),
                bool(self.show_cellpose_refs),
                bool(self.show_suite2p_iscell0.isChecked()),
                tuple(sorted(self.selected_suite2p_refs)),
                tuple(sorted(self.cellpose_final_original_ids())),
                len(self.added_rois),
                bool(self.playing and self.fast_play_no_overlays.isChecked()),
            )
        return (
            view,
            self._roi_overlay_revision,
            round(self.render_scale(), 3),
            bool(self.show_suite2p_refs),
            bool(self.show_cellpose_refs),
            bool(self.show_suite2p_iscell0.isChecked()),
            bool(self.show_final_rois.isChecked()),
            self.selected_roi,
            self.selected_manual_roi,
            tuple(sorted(self.valid_candidate_suite2p_selection())),
            tuple(sorted(self.valid_candidate_cellpose_selection())),
            tuple(sorted(self.selected_suite2p_refs)),
            len(self.added_rois),
            bool(self.playing and self.fast_play_no_overlays.isChecked()),
        )

    def static_overlay_pixmap(self, view: str, width: int, height: int) -> QPixmap:
        key = self.overlay_cache_key(view)
        cached = self._overlay_pixmap_cache.get(key)
        if cached is not None:
            return QPixmap(cached)
        overlay = QPixmap(width, height)
        overlay.fill(Qt.GlobalColor.transparent)
        if self.playing and self.fast_play_no_overlays.isChecked() and view == "edit":
            return overlay
        painter = QPainter(overlay)
        painter.scale(self.render_scale(), self.render_scale())
        if view == "reference":
            self.paint_suite2p_rois(painter, selected_only=False, highlight_selection=False)
            self.paint_cellpose_rois(painter, highlight_selection=False)
        elif view == "edit":
            self.paint_suite2p_rois(painter, selected_only=True)
        if view != "edit" or self.show_final_rois.isChecked():
            self.paint_added_rois(painter)
        painter.end()
        if len(self._overlay_pixmap_cache) > 8:
            self._overlay_pixmap_cache.clear()
        self._overlay_pixmap_cache[key] = QPixmap(overlay)
        return overlay

    def frame_pixmap(self, view: str) -> QPixmap:
        pixmap = self.base_frame_pixmap()
        painter = QPainter(pixmap)
        painter.drawPixmap(0, 0, self.static_overlay_pixmap(view, pixmap.width(), pixmap.height()))
        painter.scale(self.render_scale(), self.render_scale())
        self.paint_dynamic_selection(painter, view)
        self.paint_current_polygon(painter)
        self.paint_selection_box(painter, view)
        painter.end()
        return pixmap

    @staticmethod
    def draw_boundary_points(painter: QPainter, xpix: np.ndarray, ypix: np.ndarray) -> None:
        if xpix.size == 0 or ypix.size == 0:
            return
        points = QPolygon([QPoint(int(x), int(y)) for y, x in zip(ypix, xpix)])
        painter.drawPoints(points)

    def paint_suite2p_rois(self, painter: QPainter, selected_only: bool, highlight_selection: bool = True) -> None:
        selected_suite2p, _ = self.selected_final_sets()
        if selected_only:
            if not self.show_final_rois.isChecked():
                return
        elif not self.show_suite2p_refs:
            return
        for idx, roi in enumerate(self.roi_cache):
            if not selected_only and not self.suite2p_ref_visible(idx):
                continue
            keep = idx in self.selected_suite2p_refs
            if selected_only and not keep:
                continue
            ypix = np.asarray(roi.get("boundary_ypix", []), dtype=np.int32)
            xpix = np.asarray(roi.get("boundary_xpix", []), dtype=np.int32)
            if ypix.size == 0:
                continue
            color = QColor(0, 255, 80, 230) if keep else QColor(255, 128, 0, 120)
            if highlight_selection and not selected_only and idx in self.selected_candidate_suite2p_refs:
                color = QColor(255, 220, 0, 255)
            if highlight_selection and selected_only and (self.selected_roi == idx or idx in selected_suite2p):
                color = QColor(255, 220, 0, 255)
            pen = QPen(color)
            pen.setWidth(1)
            painter.setPen(pen)
            self.draw_boundary_points(painter, xpix, ypix)

    def paint_cellpose_rois(self, painter: QPainter, highlight_selection: bool = True) -> None:
        if not self.show_cellpose_refs:
            return
        selected = self.valid_candidate_cellpose_selection() if highlight_selection else set()
        final_cellpose_ids = self.cellpose_final_original_ids()
        for idx, roi in enumerate(self.cellpose_cache):
            if idx in final_cellpose_ids:
                continue
            ypix = np.asarray(roi.get("boundary_ypix", []), dtype=np.int32)
            xpix = np.asarray(roi.get("boundary_xpix", []), dtype=np.int32)
            if ypix.size == 0:
                continue
            color = QColor(210, 80, 255, 140)
            if idx in selected:
                color = QColor(255, 220, 0, 255)
            pen = QPen(color)
            pen.setWidth(1)
            painter.setPen(pen)
            self.draw_boundary_points(painter, xpix, ypix)

    def paint_dynamic_selection(self, painter: QPainter, view: str) -> None:
        if view == "reference":
            pen = QPen(QColor(255, 220, 0, 255))
            pen.setWidth(2)
            painter.setPen(pen)
            for idx in sorted(self.valid_candidate_suite2p_selection()):
                if 0 <= idx < len(self.roi_cache):
                    roi = self.roi_cache[idx]
                    self.draw_boundary_points(
                        painter,
                        np.asarray(roi.get("boundary_xpix", []), dtype=np.int32),
                        np.asarray(roi.get("boundary_ypix", []), dtype=np.int32),
                    )
            for idx in sorted(self.valid_candidate_cellpose_selection()):
                if 0 <= idx < len(self.cellpose_cache):
                    roi = self.cellpose_cache[idx]
                    self.draw_boundary_points(
                        painter,
                        np.asarray(roi.get("boundary_xpix", []), dtype=np.int32),
                        np.asarray(roi.get("boundary_ypix", []), dtype=np.int32),
                    )

    def paint_added_rois(self, painter: QPainter) -> None:
        _, selected_manual = self.selected_final_sets()
        pen = QPen(QColor(0, 180, 255, 230))
        pen.setWidth(1)
        painter.setPen(pen)
        for idx, roi in enumerate(self.added_rois):
            points = [QPointF(float(x), float(y)) for x, y in roi.get("points", [])]
            if len(points) < 2:
                continue
            if self.selected_manual_roi == idx or idx in selected_manual:
                selected_pen = QPen(QColor(255, 220, 0, 255))
                selected_pen.setWidth(2)
                painter.setPen(selected_pen)
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

    def paint_selection_box(self, painter: QPainter, view: str) -> None:
        if self.selection_box_start is None or self.selection_box_current is None:
            return
        if self.selection_box_target in {"suite2p_refs", "candidate_refs"} and view != "reference":
            return
        if self.selection_box_target == "final" and view != "edit":
            return
        x0, y0 = self.selection_box_start
        x1, y1 = self.selection_box_current
        corners = [
            QPointF(float(x0), float(y0)),
            QPointF(float(x1), float(y0)),
            QPointF(float(x1), float(y1)),
            QPointF(float(x0), float(y1)),
        ]
        pen = QPen(QColor(255, 220, 0, 220))
        pen.setWidth(1)
        painter.setPen(pen)
        for p0, p1 in zip(corners, corners[1:] + corners[:1]):
            painter.drawLine(p0, p1)

    def refresh(self) -> None:
        shape = (self.movie.shape[-2], self.movie.shape[-1])
        self.left_canvas.set_rendered_pixmap(self.frame_pixmap(view="reference"), shape)
        if not (self.playing and self.fast_play_single_view.isChecked()):
            self.right_canvas.set_rendered_pixmap(self.frame_pixmap(view="edit"), shape)
        kept = int(len(self.selected_suite2p_refs))
        self.status.showMessage(
            f"trial={self.paths.trial_id} | frame={self.frame_index}/{self.movie.shape[0]-1} | "
            f"suite2p={kept} | manual={len(self.added_rois)} | "
            f"removed temp={len(self.removed_rois)} | drawing points={len(self.current_polygon)}"
        )

    def roi_center_candidates(self) -> list[tuple[str, int, float, float]]:
        candidates: list[tuple[str, int, float, float]] = []
        if self.show_suite2p_refs:
            for idx, roi in enumerate(self.roi_cache):
                if not self.suite2p_ref_visible(idx):
                    continue
                xpix = np.asarray(roi.get("xpix", []), dtype=float)
                ypix = np.asarray(roi.get("ypix", []), dtype=float)
                if xpix.size:
                    candidates.append(("suite2p", idx, float(np.mean(xpix)), float(np.mean(ypix))))
        if self.show_cellpose_refs:
            final_cellpose_ids = self.cellpose_final_original_ids()
            for idx, roi in enumerate(self.cellpose_cache):
                if idx in final_cellpose_ids:
                    continue
                xpix = np.asarray(roi.get("xpix", []), dtype=float)
                ypix = np.asarray(roi.get("ypix", []), dtype=float)
                if xpix.size:
                    candidates.append(("cellpose", idx, float(np.mean(xpix)), float(np.mean(ypix))))
        for idx, roi in enumerate(self.added_rois):
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
        if self.selected_candidate_cellpose_refs:
            idx = sorted(self.selected_candidate_cellpose_refs)[-1]
            if 0 <= idx < len(self.cellpose_cache):
                roi = self.cellpose_cache[idx]
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
            if kind == "cellpose" and idx in self.selected_candidate_cellpose_refs:
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
            if idx in self.selected_suite2p_refs:
                self.select_final_entries([("suite2p", idx)])
            else:
                self.selected_candidate_suite2p_refs = {idx}
                self.selected_candidate_cellpose_refs = set()
                self.roi_table.clearSelection()
        elif kind == "cellpose":
            self.selected_roi = None
            self.selected_manual_roi = None
            self.selected_candidate_suite2p_refs = set()
            self.selected_candidate_cellpose_refs = {idx}
            self.roi_table.clearSelection()
        else:
            self.selected_roi = None
            self.selected_manual_roi = idx
            self.select_final_entries([("manual", idx)])
        self.invalidate_overlay_cache()
        self.refresh()
        self.update_trace_plot()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        key = event.key()
        modifiers = event.modifiers()
        command_modifier = modifiers & (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier
        )
        if key == Qt.Key.Key_A and command_modifier:
            self.select_all_final_rois()
            return
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
        if key == Qt.Key.Key_Z and command_modifier:
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
        self.ensure_manual_traces_current()
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
        np.save(iscell_path, iscell_to_save)

        kept_indices = np.asarray(sorted(self.selected_suite2p_refs), dtype=int)
        np.savetxt(kept_path, kept_indices, fmt="%d", delimiter=",", header="suite2p_original_id", comments="")
        np.savetxt(
            deleted_path,
            np.asarray([], dtype=int),
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
                    "roi_source": "suite2p",
                    "roi_type": "suite2p",
                    "source_roi_id": f"suite2p_{int(idx)}",
                    "manual_roi_id": -1,
                    "suite2p_original_id": int(idx),
                }
            )
            mask = self.suite2p_mask(int(idx))
            if np.any(mask):
                exclusion = self.final_roi_exclusion_mask(mask.shape, exclude_suite2p_idx=int(idx))
                neuropil = annulus_mask(mask, exclusion_mask=exclusion)
                neuropil_ypix, neuropil_xpix = np.nonzero(neuropil)
                suite2p_stat[-1].update(
                    {
                        "neuropil_ypix": neuropil_ypix.astype(np.int32),
                        "neuropil_xpix": neuropil_xpix.astype(np.int32),
                        "neuropil_npix": int(neuropil.sum()),
                        "neuropil_exclusion": "other_final_rois",
                    }
                )
                f, fneu, _, _, _, _ = self.traces_for_mask(
                    mask,
                    exclusion_mask=exclusion,
                )
                suite2p_f_rows.append(np.asarray(f, dtype=np.float32))
                suite2p_fneu_rows.append(np.asarray(fneu, dtype=np.float32))
            points = ordered_boundary_points(ypix, xpix)
            if len(points) >= 3:
                fiji_rois.append((safe_roi_name("suite2p", int(idx)), points))
            new_roi_set.append(
                {
                    "roi_id": f"suite2p_{idx}",
                    "source_roi_id": f"suite2p_{idx}",
                    "roi_source": "suite2p_reference",
                    "suite2p_original_id": int(idx),
                    "n_pixels": int(len(xpix)),
                    "x_mean": float(np.mean(xpix)) if xpix.size else None,
                    "y_mean": float(np.mean(ypix)) if ypix.size else None,
                    "points": [[float(x), float(y)] for x, y in points],
                }
            )
        trace_rows = []
        for manual_idx, roi in enumerate(self.added_rois):
            clean = {k: v for k, v in roi.items() if not k.startswith("_trace_")}
            serializable_rois.append(clean)
            if roi.get("status", "accepted") == "accepted":
                points = [(float(x), float(y)) for x, y in roi.get("points", [])]
                if len(points) >= 3:
                    mask = polygon_mask(points, self.movie.shape[-2:])
                    stat_entry = stat_dict_from_mask(mask, int(roi["manual_roi_id"]))
                    previous_suite2p_id = roi.get("suite2p_original_id")
                    roi_source = "cellpose" if roi.get("roi_source") == "cellpose" else "manual"
                    source_roi_id = (
                        f"cellpose_{int(roi.get('cellpose_original_id', roi['manual_roi_id']))}"
                        if roi_source == "cellpose"
                        else f"manual_{int(roi['manual_roi_id'])}"
                    )
                    stat_entry.update(
                        {
                            "roi_source": roi_source,
                            "roi_type": str(roi.get("roi_type", "manual")),
                            "source_roi_id": source_roi_id,
                        }
                    )
                    if roi_source == "cellpose":
                        stat_entry["cellpose_original_id"] = int(roi.get("cellpose_original_id", -1))
                    if previous_suite2p_id is not None:
                        try:
                            stat_entry["previous_suite2p_original_id"] = int(previous_suite2p_id)
                        except (TypeError, ValueError):
                            pass
                    exclusion = self.final_roi_exclusion_mask(mask.shape, exclude_manual_idx=manual_idx)
                    neuropil = annulus_mask(mask, exclusion_mask=exclusion)
                    neuropil_ypix, neuropil_xpix = np.nonzero(neuropil)
                    stat_entry.update(
                        {
                            "neuropil_ypix": neuropil_ypix.astype(np.int32),
                            "neuropil_xpix": neuropil_xpix.astype(np.int32),
                            "neuropil_npix": int(neuropil.sum()),
                            "neuropil_exclusion": "other_final_rois",
                        }
                    )
                    suite2p_stat.append(stat_entry)
                    fiji_rois.append((safe_roi_name("manual", int(roi["manual_roi_id"])), points))
                suite2p_f_rows.append(np.asarray(roi["_trace_F"], dtype=np.float32))
                suite2p_fneu_rows.append(np.asarray(roi["_trace_Fneu"], dtype=np.float32))
                new_roi_set.append(
                    {
                        **clean,
                        "roi_id": f"manual_{roi['manual_roi_id']}",
                        "roi_source": "cellpose" if roi.get("roi_source") == "cellpose" else "manual",
                        "source_roi_id": (
                            f"cellpose_{int(roi.get('cellpose_original_id', roi['manual_roi_id']))}"
                            if roi.get("roi_source") == "cellpose"
                            else f"manual_{roi['manual_roi_id']}"
                        ),
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

        if (
            len(suite2p_stat) == len(new_roi_set)
            and len(suite2p_f_rows) == len(suite2p_stat)
            and len(suite2p_fneu_rows) == len(suite2p_stat)
        ):
            order = sorted(range(len(suite2p_stat)), key=lambda i: final_roi_sort_key(suite2p_stat[i]))
            suite2p_stat = [suite2p_stat[i] for i in order]
            suite2p_f_rows = [suite2p_f_rows[i] for i in order]
            suite2p_fneu_rows = [suite2p_fneu_rows[i] for i in order]
            new_roi_set = [new_roi_set[i] for i in order]
            for final_idx, (stat_entry, roi_record) in enumerate(zip(suite2p_stat, new_roi_set), start=1):
                stat_entry["final_roi_id"] = int(final_idx)
                roi_record["final_roi_id"] = int(final_idx)
                roi_record["roi_id"] = f"roi_{final_idx:04d}"

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
            "source_iscell": str(self.paths.iscell_path),
            "manual_iscell_path": str(iscell_path),
            "selected_suite2p_indices_path": str(kept_path),
            "deleted_suite2p_indices_path": str(deleted_path),
            "manual_added_rois_path": str(additions_path),
            "manual_added_rois_note": "Legacy-compatible cache for non-suite2p final ROIs, including hand-drawn and Cellpose-derived ROIs.",
            "manual_roi_set_path": str(new_roi_set_path),
            "manual_roi_set_note": "Primary saved final ROI set used when reopening this trial.",
            "suite2p_compatible_dir": str(suite2p_compat_dir),
            "fiji_roiset_zip_path": str(fiji_roi_zip_path),
            "manual_added_roi_traces_path": str(trace_path) if trace_rows else None,
            "n_suite2p_roi": int(len(self.iscell)),
            "n_selected_suite2p_reference_roi": int(len(kept_indices)),
            "n_deleted_suite2p_reference_roi": 0,
            "n_non_suite2p_final_roi": int(len(self.added_rois)),
            "n_non_suite2p_final_accepted_roi": int(len(self.added_rois)),
            "n_manual_added_roi": int(len(self.added_rois)),
            "n_manual_added_accepted_roi": int(len(self.added_rois)),
            "n_new_roi_set": int(len(new_roi_set)),
            "n_suite2p_compatible_roi": int(len(suite2p_stat)),
            "n_fiji_roi": int(len(fiji_rois)),
            "neuropil_coeff": float(self.neuropil_coeff),
            "neuropil_mask_source": "local_annulus_excluding_other_final_rois",
            "f0_percentile": float(self.f0_percentile),
            "note": (
                "The saved manual ROI set contains all accepted final ROIs: selected suite2p "
                "reference ROIs plus non-suite2p ROIs such as Cellpose-derived or hand-drawn ROIs. "
                "Suite2p and Cellpose source files are not edited in place. "
                "Displayed and exported ROI traces use the motion-corrected movie when available, while "
                "brightness/contrast controls affect display only. Fneu uses a local annulus around each "
                "ROI and excludes pixels belonging to other accepted final ROIs."
            ),
        }
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        self.dirty = False
        if show_message:
            QMessageBox.information(self, "Saved", f"Saved manual curation to:\n{self.paths.output_dir}")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.maybe_save_before_switch(reason="closing"):
            save_last_trial_setting(self.data_root, self.movie_kind, self.paths.trial_id)
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
        trial_id = load_last_trial_setting(data_root, args.movie_kind, trial_ids) or trial_ids[0]
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
