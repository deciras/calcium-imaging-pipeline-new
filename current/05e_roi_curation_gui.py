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
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile as tf
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from PySide6.QtCore import QPoint, QPointF, QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
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
    suite2p_dir: Path
    plane0_dir: Path
    stat_path: Path
    iscell_path: Path
    f_path: Path
    fneu_path: Path
    curated_iscell_path: Path | None
    movie_path: Path
    output_dir: Path


def find_trial_paths(data_root: Path, trial_id: str | None, movie_kind: str) -> TrialPaths:
    suite2p_root = data_root / "05_suite2p_roi_detection"
    stat_paths = sorted(suite2p_root.rglob("suite2p/plane0/stat.npy"))
    if not stat_paths:
        raise FileNotFoundError(f"No suite2p stat.npy found under {suite2p_root}")

    if trial_id is None:
        stat_path = stat_paths[0]
    else:
        matches = [path for path in stat_paths if path.parent.parent.parent.name == trial_id]
        if not matches:
            raise FileNotFoundError(f"Trial not found in step 05 outputs: {trial_id}")
        stat_path = matches[0]

    plane0_dir = stat_path.parent
    suite2p_dir = plane0_dir.parent.parent
    trial_id = suite2p_dir.name
    curated = data_root / "05c_roi_quality_filter" / trial_id / f"{trial_id}_iscell_curated.npy"
    movie_path = find_movie_path(data_root, trial_id, movie_kind)
    return TrialPaths(
        trial_id=trial_id,
        suite2p_dir=suite2p_dir,
        plane0_dir=plane0_dir,
        stat_path=stat_path,
        iscell_path=plane0_dir / "iscell.npy",
        f_path=plane0_dir / "F.npy",
        fneu_path=plane0_dir / "Fneu.npy",
        curated_iscell_path=curated if curated.exists() else None,
        movie_path=movie_path,
        output_dir=data_root / STEP_NAME / trial_id,
    )


def find_movie_path(data_root: Path, trial_id: str, movie_kind: str) -> Path:
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


def load_iscell(paths: TrialPaths, n_roi: int) -> np.ndarray:
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


def annulus_mask(roi_mask: np.ndarray, inner_px: int = 3, outer_px: int = 12) -> np.ndarray:
    try:
        from scipy import ndimage as ndi
    except Exception:
        return np.zeros_like(roi_mask, dtype=bool)

    outer = ndi.binary_dilation(roi_mask, iterations=max(int(outer_px), 1))
    inner = ndi.binary_dilation(roi_mask, iterations=max(int(inner_px), 1))
    return outer & ~inner


class VideoCanvas(QLabel):
    clicked = Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(500, 500)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self._image_shape = (1, 1)
        self._pixmap_rect = QRect()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if self._pixmap_rect.width() <= 0 or self._pixmap_rect.height() <= 0:
            return
        point = event.position().toPoint()
        if not self._pixmap_rect.contains(point):
            return
        x = (point.x() - self._pixmap_rect.left()) / self._pixmap_rect.width() * self._image_shape[1]
        y = (point.y() - self._pixmap_rect.top()) / self._pixmap_rect.height() * self._image_shape[0]
        self.clicked.emit(float(x), float(y))

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
    def __init__(self, paths: TrialPaths, neuropil_coeff: float, f0_percentile: float, f0_eps: float) -> None:
        super().__init__()
        self.paths = paths
        self.neuropil_coeff = float(neuropil_coeff)
        self.f0_percentile = float(f0_percentile)
        self.f0_eps = float(f0_eps)
        self.stat = np.load(paths.stat_path, allow_pickle=True)
        self.iscell = load_iscell(paths, len(self.stat))
        self.F = self.load_trace_array(paths.f_path, len(self.stat))
        self.Fneu = self.load_trace_array(paths.fneu_path, len(self.stat))
        self.movie = movie_as_tyx(paths.movie_path)
        self.frame_index = 0
        self.selected_roi: int | None = None
        self.selected_manual_roi: int | None = None
        self.added_rois: list[dict] = []
        self.current_polygon: list[tuple[float, float]] = []
        self.low_pct = 1.0
        self.high_pct = 99.0

        self.setWindowTitle(f"ROI curation - {paths.trial_id}")
        self.left_canvas = VideoCanvas()
        self.right_canvas = VideoCanvas()
        self.left_canvas.clicked.connect(self.handle_overlay_click)
        self.right_canvas.clicked.connect(self.handle_right_click)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(max(self.movie.shape[0] - 1, 0))
        self.frame_slider.valueChanged.connect(self.set_frame)

        self.frame_spin = QSpinBox()
        self.frame_spin.setMinimum(0)
        self.frame_spin.setMaximum(max(self.movie.shape[0] - 1, 0))
        self.frame_spin.valueChanged.connect(self.set_frame)

        self.roi_list = QListWidget()
        self.roi_list.currentRowChanged.connect(self.select_roi_from_list)

        self.show_rejected = QCheckBox("show rejected")
        self.show_rejected.setChecked(False)
        self.show_rejected.stateChanged.connect(lambda _: self.rebuild_and_refresh())

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["select existing ROI", "draw polygon ROI"])
        self.mode_combo.currentIndexChanged.connect(lambda _: self.refresh())

        keep_btn = QPushButton("Keep selected")
        keep_btn.clicked.connect(lambda: self.set_selected_state(1))
        reject_btn = QPushButton("Reject selected")
        reject_btn.clicked.connect(lambda: self.set_selected_state(0))
        undo_btn = QPushButton("Undo polygon point")
        undo_btn.clicked.connect(self.undo_polygon_point)
        finish_btn = QPushButton("Finish polygon ROI")
        finish_btn.clicked.connect(self.finish_polygon_roi)
        clear_btn = QPushButton("Clear polygon")
        clear_btn.clicked.connect(self.clear_polygon)
        save_btn = QPushButton("Save manual curation")
        save_btn.clicked.connect(self.save_outputs)

        self.trace_canvas = TraceCanvas()

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        controls = QVBoxLayout()
        controls.addWidget(QLabel("ROI list"))
        controls.addWidget(self.roi_list)
        controls.addWidget(self.show_rejected)
        controls.addWidget(keep_btn)
        controls.addWidget(reject_btn)
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

        image_layout = QHBoxLayout()
        image_layout.addWidget(self.left_canvas, stretch=1)
        image_layout.addWidget(self.right_canvas, stretch=1)

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("frame"))
        slider_layout.addWidget(self.frame_slider)
        slider_layout.addWidget(self.frame_spin)

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.addLayout(image_layout)
        left_layout.addLayout(slider_layout)
        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addLayout(controls)

        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        self.populate_roi_list()
        self.refresh()
        self.update_trace_plot()

    @staticmethod
    def load_trace_array(path: Path, n_roi: int) -> np.ndarray | None:
        if not path.exists():
            return None
        arr = np.load(path, allow_pickle=True)
        if arr.ndim != 2 or arr.shape[0] != n_roi:
            return None
        return np.asarray(arr, dtype=np.float32)

    def populate_roi_list(self) -> None:
        current = self.selected_roi
        self.roi_list.blockSignals(True)
        self.roi_list.clear()
        for idx in range(len(self.stat)):
            if not self.show_rejected.isChecked() and self.iscell[idx, 0] <= 0:
                continue
            item = QListWidgetItem(f"{idx:04d} | {'keep' if self.iscell[idx, 0] > 0 else 'reject'}")
            item.setData(Qt.ItemDataRole.UserRole, idx)
            self.roi_list.addItem(item)
            if current == idx:
                self.roi_list.setCurrentItem(item)
        self.roi_list.blockSignals(False)

    def rebuild_and_refresh(self) -> None:
        self.populate_roi_list()
        self.refresh()

    def set_frame(self, value: int) -> None:
        value = int(value)
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
        self.selected_roi = int(item.data(Qt.ItemDataRole.UserRole))
        self.selected_manual_roi = None
        self.refresh()
        self.update_trace_plot()

    def handle_overlay_click(self, x: float, y: float) -> None:
        best_idx = None
        best_dist = float("inf")
        for idx, roi in enumerate(self.stat):
            if not isinstance(roi, dict):
                continue
            ypix = np.asarray(roi.get("ypix", []), dtype=float)
            xpix = np.asarray(roi.get("xpix", []), dtype=float)
            if ypix.size == 0:
                continue
            if self.iscell[idx, 0] <= 0 and not self.show_rejected.isChecked():
                continue
            dist = float(np.min((xpix - x) ** 2 + (ypix - y) ** 2))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is not None and best_dist <= 100:
            self.selected_roi = best_idx
            self.selected_manual_roi = None
            self.status.showMessage(f"Selected ROI {best_idx}")
            self.refresh()
            self.update_trace_plot()

    def handle_right_click(self, x: float, y: float) -> None:
        if self.mode_combo.currentText() != "draw polygon ROI":
            return
        self.current_polygon.append((float(x), float(y)))
        self.status.showMessage(f"Polygon point {len(self.current_polygon)}: x={x:.1f}, y={y:.1f}")
        self.refresh()

    def undo_polygon_point(self) -> None:
        if self.current_polygon:
            self.current_polygon.pop()
            self.refresh()

    def clear_polygon(self) -> None:
        self.current_polygon = []
        self.refresh()

    def finish_polygon_roi(self) -> None:
        if len(self.current_polygon) < 3:
            QMessageBox.warning(self, "Polygon ROI", "At least 3 points are needed.")
            return
        mask = polygon_mask(self.current_polygon, self.movie.shape[-2:])
        if int(mask.sum()) < 3:
            QMessageBox.warning(self, "Polygon ROI", "The polygon is too small.")
            return
        roi = self.build_manual_roi(mask)
        self.added_rois.append(roi)
        self.selected_roi = None
        self.selected_manual_roi = len(self.added_rois) - 1
        self.current_polygon = []
        self.status.showMessage(f"Added polygon ROI {roi['manual_roi_id']} with {roi['n_pixels']} pixels")
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
            "roi_type": "polygon",
            "points": [[float(x), float(y)] for x, y in self.current_polygon],
            "frame_added": int(self.frame_index),
            "n_pixels": int(mask.sum()),
            "x_mean": float(np.mean(xpix)),
            "y_mean": float(np.mean(ypix)),
            "neuropil_coeff": float(self.neuropil_coeff),
            "f0_percentile": float(self.f0_percentile),
            "f0": float(f0),
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
        if not np.any(mask):
            return np.full((self.movie.shape[0],), np.nan, dtype=np.float32)
        pixels = self.movie[:, mask]
        return np.asarray(np.nanmean(pixels, axis=1), dtype=np.float32)

    def set_selected_state(self, state: int) -> None:
        if self.selected_roi is None:
            return
        self.iscell[self.selected_roi, 0] = float(state)
        self.iscell[self.selected_roi, 1] = 1.0 if state else 0.0
        self.populate_roi_list()
        self.refresh()

    def update_trace_plot(self) -> None:
        if self.selected_manual_roi is not None and 0 <= self.selected_manual_roi < len(self.added_rois):
            roi = self.added_rois[self.selected_manual_roi]
            self.trace_canvas.plot_traces(
                f"Manual polygon ROI {roi['manual_roi_id']}",
                np.asarray(roi["_trace_F"], dtype=float),
                np.asarray(roi["_trace_Fneu"], dtype=float),
                np.asarray(roi["_trace_F_corrected"], dtype=float),
                np.asarray(roi["_trace_dff"], dtype=float),
            )
            return
        if self.selected_roi is None:
            self.trace_canvas.plot_empty("Select a ROI or draw a polygon ROI")
            return
        if self.F is None:
            self.trace_canvas.plot_empty("F.npy is missing, so suite2p traces cannot be shown")
            return
        f = np.asarray(self.F[self.selected_roi], dtype=float)
        fneu = None if self.Fneu is None else np.asarray(self.Fneu[self.selected_roi], dtype=float)
        fcorr = f.copy() if fneu is None else f - self.neuropil_coeff * fneu
        dff, _ = robust_dff(fcorr, self.f0_percentile, self.f0_eps)
        self.trace_canvas.plot_traces(f"suite2p ROI {self.selected_roi}", f, fneu, fcorr, dff)

    def frame_pixmap(self, overlay: bool) -> QPixmap:
        frame = normalize_frame(self.movie[self.frame_index], self.low_pct, self.high_pct)
        height, width = frame.shape
        qimg = QImage(frame.data, width, height, width, QImage.Format.Format_Grayscale8).copy()
        pixmap = QPixmap.fromImage(qimg)
        painter = QPainter(pixmap)
        if overlay:
            self.paint_rois(painter)
        self.paint_added_rois(painter)
        self.paint_current_polygon(painter)
        painter.end()
        return pixmap

    def paint_rois(self, painter: QPainter) -> None:
        for idx, roi in enumerate(self.stat):
            if not isinstance(roi, dict):
                continue
            keep = self.iscell[idx, 0] > 0
            if not keep and not self.show_rejected.isChecked():
                continue
            ypix = np.asarray(roi.get("ypix", []), dtype=np.int32)
            xpix = np.asarray(roi.get("xpix", []), dtype=np.int32)
            if ypix.size == 0:
                continue
            color = QColor(0, 255, 80, 210) if keep else QColor(255, 60, 60, 120)
            if self.selected_roi == idx:
                color = QColor(255, 220, 0, 255)
            pen = QPen(color)
            pen.setWidth(1)
            painter.setPen(pen)
            cx, cy, radius = mask_bounds(ypix, xpix)
            if np.isfinite(cx) and np.isfinite(cy):
                painter.drawEllipse(QPoint(int(round(cx)), int(round(cy))), int(round(radius)), int(round(radius)))

    def paint_added_rois(self, painter: QPainter) -> None:
        pen = QPen(QColor(0, 180, 255, 230))
        pen.setWidth(2)
        painter.setPen(pen)
        for idx, roi in enumerate(self.added_rois):
            points = [QPointF(float(x), float(y)) for x, y in roi.get("points", [])]
            if len(points) < 2:
                continue
            if self.selected_manual_roi == idx:
                selected_pen = QPen(QColor(255, 220, 0, 255))
                selected_pen.setWidth(3)
                painter.setPen(selected_pen)
            else:
                painter.setPen(pen)
            for p0, p1 in zip(points, points[1:] + points[:1]):
                painter.drawLine(p0, p1)

    def paint_current_polygon(self, painter: QPainter) -> None:
        if not self.current_polygon:
            return
        pen = QPen(QColor(255, 120, 0, 235))
        pen.setWidth(2)
        painter.setPen(pen)
        points = [QPointF(float(x), float(y)) for x, y in self.current_polygon]
        for point in points:
            painter.drawEllipse(point, 2.5, 2.5)
        for p0, p1 in zip(points, points[1:]):
            painter.drawLine(p0, p1)

    def refresh(self) -> None:
        shape = (self.movie.shape[-2], self.movie.shape[-1])
        self.left_canvas.set_rendered_pixmap(self.frame_pixmap(overlay=True), shape)
        self.right_canvas.set_rendered_pixmap(self.frame_pixmap(overlay=False), shape)
        kept = int(np.sum(self.iscell[:, 0] > 0))
        self.status.showMessage(
            f"trial={self.paths.trial_id} | frame={self.frame_index}/{self.movie.shape[0]-1} | "
            f"kept={kept}/{len(self.iscell)} | added={len(self.added_rois)} | polygon points={len(self.current_polygon)}"
        )

    def save_outputs(self) -> None:
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        iscell_path = self.paths.output_dir / f"{self.paths.trial_id}_iscell_manual.npy"
        additions_path = self.paths.output_dir / f"{self.paths.trial_id}_manual_added_rois.json"
        trace_path = self.paths.output_dir / f"{self.paths.trial_id}_manual_added_roi_traces.csv"
        summary_path = self.paths.output_dir / f"{self.paths.trial_id}_manual_curation_summary.json"
        np.save(iscell_path, self.iscell.astype(np.float32))
        serializable_rois = []
        trace_rows = []
        for roi in self.added_rois:
            clean = {k: v for k, v in roi.items() if not k.startswith("_trace_")}
            serializable_rois.append(clean)
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
        if trace_rows:
            with trace_path.open("w", encoding="utf-8") as handle:
                handle.write("manual_roi_id,frame,F,Fneu,F_corrected,dff\n")
                for row in trace_rows:
                    handle.write(",".join(str(value) for value in row) + "\n")
        summary = {
            "trial_id": self.paths.trial_id,
            "movie_path": str(self.paths.movie_path),
            "source_iscell": str(self.paths.curated_iscell_path or self.paths.iscell_path),
            "manual_iscell_path": str(iscell_path),
            "manual_added_rois_path": str(additions_path),
            "manual_added_roi_traces_path": str(trace_path) if trace_rows else None,
            "n_suite2p_roi": int(len(self.iscell)),
            "n_kept_existing_roi": int(np.sum(self.iscell[:, 0] > 0)),
            "n_rejected_existing_roi": int(np.sum(self.iscell[:, 0] <= 0)),
            "n_manual_added_roi": int(len(self.added_rois)),
            "neuropil_coeff": float(self.neuropil_coeff),
            "f0_percentile": float(self.f0_percentile),
            "note": "Manual polygon ROI neuropil is estimated from a local annulus in the selected movie.",
        }
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        QMessageBox.information(self, "Saved", f"Saved manual curation to:\n{self.paths.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open a local ROI curation GUI.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--trial-id", help="Trial folder name. Defaults to the first available suite2p trial.")
    parser.add_argument("--movie-kind", choices=("corrected", "spatial-highpass"), default="corrected")
    parser.add_argument("--neuropil-coeff", type=float, default=0.7)
    parser.add_argument("--f0-percentile", type=float, default=10.0)
    parser.add_argument("--f0-eps", type=float, default=1e-6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = find_trial_paths(args.data_root.expanduser().resolve(), args.trial_id, args.movie_kind)
    app = QApplication(sys.argv[:1])
    window = CurationWindow(
        paths,
        neuropil_coeff=args.neuropil_coeff,
        f0_percentile=args.f0_percentile,
        f0_eps=args.f0_eps,
    )
    window.resize(1550, 900)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
