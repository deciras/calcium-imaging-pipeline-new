#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight Qt dashboard for launching and monitoring run_pipeline.py.

This GUI is intentionally a controller, not a compute host: heavy steps still run
in a child process so the Qt event loop stays responsive.
"""

from __future__ import annotations

import os
import platform
import re
import signal
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QSettings, QTimer, Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


STEP_ROWS: tuple[tuple[str, str], ...] = (
    ("00", "organize OIR files"),
    ("01", "Fiji OIR to TIF"),
    ("02", "generate stimulus map"),
    ("03", "CaImAn motion correction"),
    ("04", "spatial high-pass"),
    ("05", "suite2p ROI detection"),
    ("cellpose", "cellpose ROI segmentation"),
    ("manual", "manual ROI curation GUI"),
    ("06", "extract dF/F"),
    ("07", "detect calcium events"),
    ("08", "stimulus response analysis"),
    ("09", "angle tuning analysis"),
    ("10", "plot ROI traces"),
    ("11", "population features"),
    ("12", "stimulus-slice features"),
    ("13", "population similarity"),
    ("14", "hierarchical clustering"),
    ("15", "Leiden community detection"),
    ("16", "dimensionality reduction"),
    ("17", "cross-trial summary"),
    ("18", "report generation"),
)

STEP_GROUPS: tuple[tuple[str, str], ...] = (
    ("premanual", "Pre-manual processing and ROI candidates (00-05 + cellpose)"),
    ("manual", "Manual ROI curation (manual)"),
    ("postmanual", "Post-manual analysis and reports (06-18)"),
    ("basic-analysis", "Basic response analysis (06-09)"),
    ("core-analysis", "Core tuning and clustering analysis (06,08,09,10-14)"),
    ("custom", "Custom selection (checked steps below)"),
)


class PipelineDashboard(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.repo_root = Path(__file__).resolve().parents[1]
        self.current_dir = self.repo_root / "current"
        self.pipeline_script = self.current_dir / "run_pipeline.py"
        self.settings = QSettings("calcium-imaging-pipeline", "pipeline-dashboard")
        self.process: QProcess | None = None
        self.background_session_name: str | None = None
        self.background_log_path: Path | None = None
        self.background_status_path: Path | None = None
        self.background_log_offset: int = 0
        self.background_poll_counter: int = 0
        self._buffer = ""
        self.step_state: dict[str, str] = {step: "idle" for step, _ in STEP_ROWS}
        self.step_message: dict[str, str] = {step: "" for step, _ in STEP_ROWS}
        self.background_poll_timer = QTimer(self)
        self.background_poll_timer.setInterval(1500)
        self.background_poll_timer.timeout.connect(self.poll_background_run)

        self.setWindowTitle("Calcium pipeline dashboard")
        self.resize(1180, 780)
        self._build_ui()
        self.refresh_step_table()
        self.update_button_state(False)

    def _build_ui(self) -> None:
        self.data_root_edit = QLineEdit(self.default_data_root())
        browse_data_btn = QPushButton("Browse")
        browse_data_btn.clicked.connect(self.browse_data_root)

        self.conda_bin_edit = QLineEdit(os.environ.get("CONDA_BIN", "conda"))
        self.fiji_bin_edit = QLineEdit(self.default_fiji_path())
        browse_fiji_btn = QPushButton("Browse")
        browse_fiji_btn.clicked.connect(self.browse_fiji_bin)
        save_fiji_default_btn = QPushButton("Set default")
        save_fiji_default_btn.clicked.connect(self.save_fiji_default)

        self.step_group_combo = QComboBox()
        for value, label in STEP_GROUPS:
            self.step_group_combo.addItem(label, value)
        self.step_group_combo.currentIndexChanged.connect(self.sync_step_group)

        self.step_list = QListWidget()
        self.step_list.setMaximumHeight(180)
        for step_id, name in STEP_ROWS:
            item = QListWidgetItem(f"{step_id}  {name}")
            item.setData(Qt.ItemDataRole.UserRole, step_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.step_list.addItem(item)
        self.step_list.itemChanged.connect(self.mark_custom_group)

        self.action_combo = QComboBox()
        self.action_combo.addItems(["skip", "overwrite"])
        self.analysis_source_combo = QComboBox()
        self.analysis_source_combo.addItem("Stimulus slices (recommended; steps 08/12)", "slices")
        self.analysis_source_combo.addItem("Full traces (complete dF/F time series)", "traces")
        self.analysis_source_combo.addItem("Summary features (scalar ROI metrics)", "features")
        self.analysis_source_combo.addItem("Response scalars only", "responses")
        self.cluster_normalization_combo = QComboBox()
        self.cluster_normalization_combo.addItem("Normalized clustering (default)", "normalized")
        self.cluster_normalization_combo.addItem("Raw/source-scale clustering", "raw")
        self.cluster_normalization_combo.addItem("Both normalized and raw/source-scale", "both")
        self.cluster_normalization_combo.setCurrentIndex(2)
        self.background_run_check = QCheckBox("Background run (survive lock screen)")
        self.background_run_check.setChecked(platform.system() == "Linux")
        self.step_dry_run_check = QCheckBox("step dry-run")
        self.step_dry_run_check.setChecked(False)

        self.fiji_memory_edit = QLineEdit(os.environ.get("FIJI_MEMORY", self.default_fiji_memory()))
        self.suite2p_threads_spin = self.integer_spin(1, 128, int(os.environ.get("SUITE2P_THREADS", "8")))
        self.n_workers_spin = self.integer_spin(1, 64, int(os.environ.get("N_WORKERS", "1")))
        self.num_threads_spin = self.integer_spin(1, 128, int(os.environ.get("NUM_THREADS", "1")))
        self.trial_id_edit = QLineEdit("")
        self.extra_args_edit = QLineEdit("")

        self.command_preview = QPlainTextEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setMaximumHeight(90)

        plan_btn = QPushButton("Plan")
        plan_btn.clicked.connect(lambda: self.start_pipeline(plan_only=True))
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(lambda: self.start_pipeline(plan_only=False))
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_pipeline)
        manual_btn = QPushButton("Open manual GUI")
        manual_btn.clicked.connect(self.open_manual_gui)
        clear_log_btn = QPushButton("Clear log")
        clear_log_btn.clicked.connect(self.clear_log)

        data_row = QHBoxLayout()
        data_row.addWidget(self.data_root_edit)
        data_row.addWidget(browse_data_btn)
        fiji_row = QHBoxLayout()
        fiji_row.addWidget(self.fiji_bin_edit)
        fiji_row.addWidget(browse_fiji_btn)
        fiji_row.addWidget(save_fiji_default_btn)

        config_form = QFormLayout()
        config_form.addRow("Data root", data_row)
        config_form.addRow("Conda", self.conda_bin_edit)
        config_form.addRow("Fiji", fiji_row)
        config_form.addRow("Step group", self.step_group_combo)
        config_form.addRow("Action", self.action_combo)
        config_form.addRow("Analysis input", self.analysis_source_combo)
        config_form.addRow("Cluster scaling", self.cluster_normalization_combo)
        config_form.addRow("Fiji memory", self.fiji_memory_edit)
        config_form.addRow("suite2p threads", self.suite2p_threads_spin)
        config_form.addRow("workers", self.n_workers_spin)
        config_form.addRow("threads / worker", self.num_threads_spin)
        config_form.addRow("Trial id", self.trial_id_edit)
        config_form.addRow("Extra args", self.extra_args_edit)

        config_box = QGroupBox("Run configuration")
        config_box.setLayout(config_form)

        button_row = QHBoxLayout()
        button_row.addWidget(plan_btn)
        button_row.addWidget(self.start_btn)
        button_row.addWidget(self.stop_btn)
        button_row.addWidget(manual_btn)
        button_row.addWidget(clear_log_btn)
        button_row.addWidget(self.background_run_check)
        button_row.addWidget(self.step_dry_run_check)
        button_row.addStretch(1)

        left_layout = QVBoxLayout()
        left_layout.addWidget(config_box)
        left_layout.addWidget(QLabel("Steps"))
        left_layout.addWidget(self.step_list)
        left_layout.addWidget(QLabel("Command"))
        left_layout.addWidget(self.command_preview)
        left_layout.addLayout(button_row)

        self.step_table = QTableWidget(0, 3)
        self.step_table.setHorizontalHeaderLabels(["step", "state", "last message"])
        self.step_table.verticalHeader().setVisible(False)
        self.step_table.horizontalHeader().setStretchLastSection(True)
        self.step_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(6000)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Status"))
        right_layout.addWidget(self.step_table, stretch=1)
        right_layout.addWidget(QLabel("Log"))
        right_layout.addWidget(self.log_view, stretch=2)

        main_layout = QGridLayout()
        main_layout.addLayout(left_layout, 0, 0)
        main_layout.addLayout(right_layout, 0, 1)
        main_layout.setColumnStretch(0, 1)
        main_layout.setColumnStretch(1, 2)

        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        self.sync_step_group()
        self.update_command_preview()
        for control in (
            self.data_root_edit,
            self.conda_bin_edit,
            self.fiji_bin_edit,
            self.fiji_memory_edit,
            self.trial_id_edit,
            self.extra_args_edit,
        ):
            control.textChanged.connect(self.update_command_preview)
        for control in (self.action_combo, self.analysis_source_combo, self.cluster_normalization_combo, self.suite2p_threads_spin, self.n_workers_spin, self.num_threads_spin):
            if hasattr(control, "currentIndexChanged"):
                control.currentIndexChanged.connect(self.update_command_preview)
            else:
                control.valueChanged.connect(self.update_command_preview)
        self.step_dry_run_check.stateChanged.connect(self.update_command_preview)

    @staticmethod
    def integer_spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def default_data_root(self) -> str:
        configured = os.environ.get("DATA_ROOT", "").strip()
        if configured:
            return configured
        return self.settings.value("last_data_root", "", type=str)

    def save_data_root_default(self) -> None:
        data_root = self.data_root_edit.text().strip()
        if not data_root:
            return
        self.settings.setValue("last_data_root", data_root)
        self.settings.sync()

    def default_fiji_path(self) -> str:
        saved = self.settings.value("fiji_path", "", type=str)
        if saved:
            return saved
        configured = os.environ.get("FIJI_BIN") or os.environ.get("FIJI_PATH")
        if configured:
            return configured
        if platform.system() == "Darwin":
            return "/Applications/Fiji.app"
        if platform.system() == "Linux":
            return "/home/yifei/Fiji/fiji-linux-x64"
        return ""

    @staticmethod
    def default_fiji_memory() -> str:
        if platform.system() == "Linux":
            return "64G"
        return "16G"

    def browse_data_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select data root", self.data_root_edit.text() or str(Path.home()))
        if path:
            self.data_root_edit.setText(path)
            self.save_data_root_default()

    def browse_fiji_bin(self) -> None:
        current = self.fiji_bin_edit.text() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Select Fiji app or directory", current)
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Select Fiji executable", current)
        if path:
            self.fiji_bin_edit.setText(path)

    def save_fiji_default(self) -> None:
        path = self.fiji_bin_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Fiji default", "Set a Fiji path before saving it as the default.")
            return
        self.settings.setValue("fiji_path", path)
        self.settings.sync()
        self.append_log(f"Saved Fiji default: {path}\n")

    def selected_steps_text(self) -> str:
        selected: list[str] = []
        for row in range(self.step_list.count()):
            item = self.step_list.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                selected.append(str(item.data(Qt.ItemDataRole.UserRole)))
        return ",".join(selected)

    def sync_step_group(self) -> None:
        value = self.step_group_combo.currentData()
        if value == "custom":
            self.update_command_preview()
            return
        group_steps = {
            "premanual": {"00", "01", "02", "03", "04", "05", "cellpose"},
            "manual": {"manual"},
            "postmanual": {"06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18"},
            "basic-analysis": {"06", "07", "08", "09"},
            "core-analysis": {"06", "08", "09", "10", "11", "12", "13", "14"},
        }.get(str(value), set())
        self.step_list.blockSignals(True)
        for row in range(self.step_list.count()):
            item = self.step_list.item(row)
            item.setCheckState(
                Qt.CheckState.Checked
                if str(item.data(Qt.ItemDataRole.UserRole)) in group_steps
                else Qt.CheckState.Unchecked
            )
        self.step_list.blockSignals(False)
        self.update_command_preview()

    def mark_custom_group(self) -> None:
        custom_idx = self.step_group_combo.findData("custom")
        if custom_idx >= 0 and self.step_group_combo.currentIndex() != custom_idx:
            self.step_group_combo.blockSignals(True)
            self.step_group_combo.setCurrentIndex(custom_idx)
            self.step_group_combo.blockSignals(False)
        self.update_command_preview()

    def build_command(self, plan_only: bool = False, manual_only: bool = False) -> tuple[str, list[str]]:
        program = sys.executable
        args = [str(self.pipeline_script)]
        steps = "manual" if manual_only else self.selected_steps_text()
        if not steps:
            raise ValueError("No pipeline steps are selected.")
        args.extend(["--steps", steps])
        data_root = self.data_root_edit.text().strip()
        if data_root:
            args.extend(["--data-root", data_root])
        args.extend(["--action", self.action_combo.currentText()])
        source = str(self.analysis_source_combo.currentData())
        args.extend(["--similarity-source", source])
        args.extend(["--cluster-source", source])
        if source != "angle":
            args.extend(["--embedding-source", "features" if source == "angle" else source])
        args.extend(["--cluster-normalization", str(self.cluster_normalization_combo.currentData())])
        conda_bin = self.conda_bin_edit.text().strip()
        if conda_bin:
            args.extend(["--conda-bin", conda_bin])
        if plan_only:
            args.append("--dry-run")
        if self.step_dry_run_check.isChecked():
            args.append("--step-dry-run")
        fiji_memory = self.fiji_memory_edit.text().strip()
        if fiji_memory:
            args.extend(["--fiji-memory", fiji_memory])
        args.extend(["--suite2p-threads", str(self.suite2p_threads_spin.value())])
        args.extend(["--n-workers", str(self.n_workers_spin.value())])
        args.extend(["--num-threads", str(self.num_threads_spin.value())])
        trial_id = self.trial_id_edit.text().strip()
        if trial_id:
            args.extend(["--trial-id", trial_id])
        extra = self.extra_args_edit.text().strip()
        if extra:
            args.extend(shlex.split(extra))
        return program, args

    def build_environment_map(self) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        data_root = self.data_root_edit.text().strip()
        if data_root:
            env["DATA_ROOT"] = data_root
            env["PIPELINE_TMPDIR"] = str(Path(data_root) / ".tmp")
            env["PIPELINE_CACHE_DIR"] = str(Path(data_root) / ".cache")
        fiji_bin = self.fiji_bin_edit.text().strip()
        if fiji_bin:
            env["FIJI_BIN"] = fiji_bin
        env["FIJI_MEMORY"] = self.fiji_memory_edit.text().strip() or self.default_fiji_memory()
        env["SUITE2P_THREADS"] = str(self.suite2p_threads_spin.value())
        env["N_WORKERS"] = str(self.n_workers_spin.value())
        env["NUM_THREADS"] = str(self.num_threads_spin.value())
        return env

    def process_environment(self) -> QProcessEnvironment:
        env = QProcessEnvironment()
        for key, value in self.build_environment_map().items():
            env.insert(key, value)
        return env

    def update_command_preview(self) -> None:
        try:
            program, args = self.build_command(plan_only=False)
            text = subprocess_like_command([program, *args])
        except Exception as exc:
            text = f"Cannot build command: {exc}"
        self.command_preview.setPlainText(text)

    def start_pipeline(self, plan_only: bool = False) -> None:
        if self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, "Pipeline running", "A pipeline process is already running.")
            return
        if self.background_session_name is not None:
            QMessageBox.warning(self, "Pipeline running", "A pipeline process is already running.")
            return
        try:
            program, args = self.build_command(plan_only=plan_only)
        except Exception as exc:
            QMessageBox.warning(self, "Pipeline command", str(exc))
            return
        self.save_data_root_default()
        self.reset_step_states()
        self.append_log(f"$ {subprocess_like_command([program, *args])}\n")
        if self.background_run_check.isChecked() and not plan_only and platform.system() == "Linux":
            self.start_background_pipeline(program, args)
            return
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(self.current_dir))
        self.process.setProcessEnvironment(self.process_environment())
        self.process.setProgram(program)
        self.process.setArguments(args)
        self.process.readyReadStandardOutput.connect(self.read_stdout)
        self.process.readyReadStandardError.connect(self.read_stderr)
        self.process.finished.connect(self.process_finished)
        self.process.start()
        if not self.process.waitForStarted(3000):
            QMessageBox.warning(self, "Pipeline start", self.process.errorString())
            self.process = None
            return
        self.update_button_state(True)

    def open_manual_gui(self) -> None:
        try:
            program, args = self.build_command(plan_only=False, manual_only=True)
        except Exception as exc:
            QMessageBox.warning(self, "Manual GUI", str(exc))
            return
        self.save_data_root_default()
        ok = QProcess.startDetached(program, args, str(self.current_dir))
        if ok:
            self.append_log(f"$ {subprocess_like_command([program, *args])}\n")
            self.append_log("Manual ROI GUI launched as a separate process.\n")
        else:
            QMessageBox.warning(self, "Manual GUI", "Could not launch manual ROI GUI.")

    def stop_pipeline(self) -> None:
        if self.background_session_name is not None:
            self.append_log(f"\nStopping background tmux session {self.background_session_name}...\n")
            subprocess.run(
                ["tmux", "kill-session", "-t", self.background_session_name],
                check=False,
                capture_output=True,
                text=True,
            )
            self.finish_background_run(exit_code=15, status_text="stopped")
            return
        if self.process is None or self.process.state() == QProcess.ProcessState.NotRunning:
            return
        self.append_log("\nStopping pipeline process...\n")
        root_pid = int(self.process.processId())
        self.terminate_process_tree(root_pid, signal.SIGTERM)
        if not self.process.waitForFinished(5000):
            self.append_log("Process tree did not terminate; killing it.\n")
            self.terminate_process_tree(root_pid, signal.SIGKILL)
            self.process.kill()

    @staticmethod
    def child_pids(parent_pid: int) -> list[int]:
        try:
            output = subprocess.check_output(["ps", "-axo", "pid=,ppid="], text=True)
        except Exception:
            return []
        children: list[int] = []
        for line in output.splitlines():
            fields = line.split()
            if len(fields) != 2:
                continue
            pid, ppid = (int(fields[0]), int(fields[1]))
            if ppid == parent_pid:
                children.append(pid)
        return children

    @classmethod
    def descendant_pids(cls, root_pid: int) -> list[int]:
        descendants: list[int] = []
        stack = cls.child_pids(root_pid)
        while stack:
            pid = stack.pop()
            descendants.append(pid)
            stack.extend(cls.child_pids(pid))
        return descendants

    @classmethod
    def terminate_process_tree(cls, root_pid: int, sig: signal.Signals) -> None:
        for pid in reversed(cls.descendant_pids(root_pid)):
            try:
                os.kill(pid, sig)
            except OSError:
                pass
        try:
            os.kill(root_pid, sig)
        except OSError:
            pass

    def clear_log(self) -> None:
        self.log_view.clear()

    def read_stdout(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.consume_output(text)

    def read_stderr(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        self.consume_output(text)

    def consume_output(self, text: str) -> None:
        self.append_log(text)
        self._buffer += text
        lines = self._buffer.splitlines(keepends=True)
        self._buffer = ""
        for line in lines:
            if line.endswith("\n") or line.endswith("\r"):
                self.parse_log_line(line.strip())
            else:
                self._buffer = line
        self.refresh_step_table()

    def parse_log_line(self, line: str) -> None:
        start_match = re.search(r"Starting step ([A-Za-z0-9_-]+) (.+)", line)
        if start_match:
            step_id = start_match.group(1)
            self.step_state[step_id] = "running"
            self.step_message[step_id] = start_match.group(2)
            return
        success_match = re.search(r"Step ([A-Za-z0-9_-]+) .+ finished successfully", line)
        if success_match:
            step_id = success_match.group(1)
            self.step_state[step_id] = "done"
            self.step_message[step_id] = line
            return
        failed_match = re.search(r"Step ([A-Za-z0-9_-]+) .+ failed", line)
        if failed_match:
            step_id = failed_match.group(1)
            self.step_state[step_id] = "failed"
            self.step_message[step_id] = line
            return
        ok_match = re.search(r"\[ok\]\s+(.+)$", line)
        if ok_match:
            running = [step for step, state in self.step_state.items() if state == "running"]
            if running:
                self.step_message[running[-1]] = f"ok: {ok_match.group(1)}"

    def process_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self.append_log(f"\nProcess finished with exit code {exit_code}.\n")
        if exit_code != 0:
            for step, state in list(self.step_state.items()):
                if state == "running":
                    self.step_state[step] = "failed"
                    self.step_message[step] = f"process exited with code {exit_code}"
        self.refresh_step_table()
        self.update_button_state(False)

    def background_log_root(self) -> Path:
        data_root = self.data_root_edit.text().strip()
        if data_root:
            return Path(data_root) / "pipeline_logs"
        return self.repo_root / "pipeline_logs"

    def start_background_pipeline(self, program: str, args: list[str]) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"calcium-pipeline-{timestamp}"
        log_root = self.background_log_root()
        log_root.mkdir(parents=True, exist_ok=True)
        log_path = log_root / f"{session_name}.log"
        status_path = log_root / f"{session_name}.exitcode"
        env_map = self.build_environment_map()
        command_text = subprocess_like_command([program, *args])
        shell_command = (
            f"cd {shlex.quote(str(self.current_dir))} && "
            f"{command_text} >> {shlex.quote(str(log_path))} 2>&1; "
            f"status=$?; "
            f"printf '%s\\n' \"$status\" > {shlex.quote(str(status_path))}"
        )
        if status_path.exists():
            status_path.unlink()
        completed = subprocess.run(
            ["tmux", "new-session", "-d", "-s", session_name, "/bin/bash", "-lc", shell_command],
            capture_output=True,
            text=True,
            env=env_map,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip() or completed.stdout.strip() or "Could not launch background run."
            QMessageBox.warning(self, "Pipeline start", stderr)
            return
        self.background_session_name = session_name
        self.background_log_path = log_path
        self.background_status_path = status_path
        self.background_log_offset = 0
        self.background_poll_counter = 0
        self.append_log(f"Background tmux session: {session_name}\n")
        self.append_log(f"Background log: {log_path}\n")
        self.background_poll_timer.start()
        self.update_button_state(True)

    def read_background_log_increment(self) -> None:
        if self.background_log_path is None or not self.background_log_path.exists():
            return
        with self.background_log_path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(self.background_log_offset)
            text = handle.read()
            self.background_log_offset = handle.tell()
        if text:
            self.consume_output(text)

    def poll_background_run(self) -> None:
        if self.background_session_name is None:
            self.background_poll_timer.stop()
            return
        self.read_background_log_increment()
        self.background_poll_counter += 1
        if self.background_poll_counter % 2 != 0:
            return
        completed = subprocess.run(
            ["tmux", "has-session", "-t", self.background_session_name],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            return
        exit_code = 1
        status_text = "finished"
        if self.background_status_path is not None and self.background_status_path.exists():
            try:
                exit_code = int(self.background_status_path.read_text(encoding="utf-8").strip() or "1")
            except ValueError:
                exit_code = 1
                status_text = "finished-invalid-exitcode"
        else:
            status_text = "finished-no-exitcode"
        self.finish_background_run(exit_code=exit_code, status_text=status_text)

    def finish_background_run(self, exit_code: int, status_text: str) -> None:
        self.background_poll_timer.stop()
        self.read_background_log_increment()
        self.background_session_name = None
        self.background_log_path = None
        self.background_status_path = None
        self.background_log_offset = 0
        self.background_poll_counter = 0
        self.append_log(f"\nBackground pipeline finished with exit code {exit_code} ({status_text}).\n")
        if exit_code != 0:
            for step, state in list(self.step_state.items()):
                if state == "running":
                    self.step_state[step] = "failed"
                    self.step_message[step] = f"background run exited with code {exit_code}"
        self.refresh_step_table()
        self.update_button_state(False)

    def append_log(self, text: str) -> None:
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)
        self.log_view.insertPlainText(text)
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)

    def reset_step_states(self) -> None:
        selected = set(self.selected_steps_text().split(",")) if self.selected_steps_text() else set()
        for step_id, _ in STEP_ROWS:
            self.step_state[step_id] = "queued" if step_id in selected else "idle"
            self.step_message[step_id] = ""
        self.refresh_step_table()

    def refresh_step_table(self) -> None:
        self.step_table.setRowCount(0)
        for step_id, name in STEP_ROWS:
            row = self.step_table.rowCount()
            self.step_table.insertRow(row)
            values = [f"{step_id} {name}", self.step_state.get(step_id, "idle"), self.step_message.get(step_id, "")]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                self.step_table.setItem(row, col, item)
        self.step_table.resizeColumnsToContents()

    def update_button_state(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)


def subprocess_like_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def main() -> int:
    app = QApplication(sys.argv[:1])
    window = PipelineDashboard()
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
